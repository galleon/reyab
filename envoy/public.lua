-- Public listener (VS Code). Responsibilities:
--   1. authenticate the team key
--   2. read "model" from the body and pick the backend (or ask the router for "auto")
--   3. mark chat turns for capture; the route mirrors the request to the recorder
--      (which shadows it to Opus) and envoy_on_response streams the primary
--      answer through untouched while copying it to the recorder.
local C = dofile("/etc/envoy/common.lua")

function envoy_on_request(h)
  C.strip_internal_headers(h)
  local path = h:headers():get(":path") or ""

  -- The voting page itself loads without a key; its API calls send one.
  if path == "/arena" or path:find("^/arena%?") then return end

  local team = C.authenticate(h)
  if team == nil then
    C.json_error(h, 401, "invalid or missing API key")
    return
  end
  h:headers():remove("authorization")       -- never forward team keys upstream
  h:headers():add("x-bench-team", team)

  if path:find("^/arena/") then return end   -- voting API: auth only, no model routing
  if h:headers():get(":method") ~= "POST" then return end

  local body = h:body()
  if body == nil or body:length() == 0 then return end
  local raw = body:getBytes(0, body:length())

  local model = C.extract_model(raw, C.PUBLIC_MODELS)
  if model == nil then
    C.json_error(h, 400, "unknown model; available: " .. C.model_list())
    return
  end

  local backend = C.PUBLIC_MODELS[model]
  local route_reason = "user-selected"
  if backend == "auto" then
    if C.AUTO_ROUTER == "nvidia" or not path:find("^/v1/chat/completions") then
      backend, route_reason = C.route_auto(h, raw)
    else
      -- The ext_proc filter (vLLM Semantic Router) decides; see envoy.yaml.
      backend, route_reason = "vsr", "vsr"
    end
  end
  h:headers():add("x-bench-backend", backend)
  h:headers():add("x-bench-model", model)
  h:headers():add("x-bench-route", route_reason)

  if path:find("^/v1/chat/completions") then
    local md = h:streamInfo():dynamicMetadata()
    md:set("bench", "capture", "1")
    md:set("bench", "rid", h:headers():get("x-request-id") or "")
    md:set("bench", "team", team)
    md:set("bench", "model", model)
    md:set("bench", "backend", backend)
  end

  h:clearRouteCache()
end

function envoy_on_response(h)
  local md = h:streamInfo():dynamicMetadata():get("bench")
  if md == nil or md["capture"] ~= "1" then return end

  local status = h:headers():get(":status") or "0"
  local backend, route = md["backend"], nil
  if backend == "vsr" then
    -- The semantic router reports its choice on the response.
    local chosen = h:headers():get("x-vsr-selected-model")
    backend = C.backend_for(chosen) or "oss-a"
    local decision = h:headers():get("x-vsr-selected-decision")
    route = chosen and ("vsr:" .. (decision ~= nil and decision ~= "" and decision or "default")) or "vsr-fallback"
  end
  -- bodyChunks() yields chunks as they arrive without buffering, so SSE
  -- streaming to VS Code is not delayed.
  local parts = {}
  for chunk in h:bodyChunks() do
    local n = chunk:length()
    if n > 0 then parts[#parts + 1] = chunk:getBytes(0, n) end
  end

  h:httpCall("recorder", {
    [":method"] = "POST", [":path"] = "/ingest/primary", [":authority"] = "recorder",
    ["content-type"] = "application/octet-stream",
    ["x-request-id"] = md["rid"], ["x-bench-team"] = md["team"],
    ["x-bench-model"] = md["model"], ["x-bench-backend"] = backend,
    ["x-bench-route"] = route or "", ["x-bench-status"] = status,
  }, table.concat(parts), 10000, true)          -- async: fire and forget
end
