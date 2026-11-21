-- Shared helpers for the agentrouter Lua filters.
-- Placeholders (__X__) are substituted by render.sh at container start.
local M = {}

M.OSS_A = "__OSS_A_NAME__"
M.OSS_B = "__OSS_B_NAME__"
M.SHADOW_MODEL = "__SHADOW_MODEL__"
M.ENABLE_AUTO = ("__ENABLE_AUTO__" == "1")
-- Who decides for "auto": "vsr" (vLLM Semantic Router via ext_proc, default)
-- or "nvidia" (the classifier service's /route endpoint).
M.AUTO_ROUTER = "__AUTO_ROUTER__"

-- Models a VS Code client may request -> backend label (matches route headers).
M.PUBLIC_MODELS = { [M.OSS_A] = "oss-a", [M.OSS_B] = "oss-b" }
if M.ENABLE_AUTO then M.PUBLIC_MODELS["auto"] = "auto" end

-- ---------------------------------------------------------------- auth ----
local TEAMS = nil
do
  local ok, t = pcall(dofile, "/etc/envoy/teams.lua")
  if ok and type(t) == "table" and next(t) ~= nil then TEAMS = t end
end

-- Returns the team id, or nil when a key is required and missing/invalid.
function M.authenticate(h)
  if TEAMS == nil then return "anon" end
  local auth = h:headers():get("authorization") or ""
  local key = auth:match("^[Bb]earer%s+(.+)$")
  if key == nil then return nil end
  return TEAMS[key]
end

-- ------------------------------------------------------------- helpers ----
local SPOOFABLE = { "x-bench-backend", "x-bench-team", "x-bench-model", "x-bench-route",
                    "x-selected-model", "x-vsr-skip-processing", "x-vsr-debug" }
function M.strip_internal_headers(h)
  for _, k in ipairs(SPOOFABLE) do h:headers():remove(k) end
end

function M.json_error(h, status, msg)
  local escaped = msg:gsub('\\', '\\\\'):gsub('"', '\\"')
  h:respond({ [":status"] = tostring(status), ["content-type"] = "application/json" },
    '{"error":{"message":"' .. escaped .. '","type":"invalid_request_error"}}')
end

-- Find the top-level "model" value without a JSON parser: take the first
-- unescaped `"model": "<v>"` whose value is an allowed model. Occurrences
-- inside message text are escaped (\"model\") and skipped.
function M.extract_model(raw, allowed)
  local init = 1
  while true do
    local s, e, val = raw:find('"model"%s*:%s*"([^"\\]*)"', init)
    if s == nil then return nil end
    if (s == 1 or raw:sub(s - 1, s - 1) ~= "\\") and allowed[val] ~= nil then
      return val
    end
    init = e + 1
  end
end

-- Map a model name chosen by the semantic router back to a backend label.
function M.backend_for(model)
  if model == M.OSS_A then return "oss-a" end
  if model == M.OSS_B then return "oss-b" end
  return nil
end

-- Ask the NVIDIA-classifier router which OSS backend should serve "auto".
function M.route_auto(h, raw)
  local ok, headers, body = pcall(function()
    return h:httpCall("router", {
      [":method"] = "POST", [":path"] = "/route", [":authority"] = "router",
      ["content-type"] = "application/json",
    }, raw, 3000, false)
  end)
  if ok and body ~= nil then
    local b = body:match('"backend"%s*:%s*"(oss%-[ab])"')
    local reason = body:match('"reason"%s*:%s*"([^"]*)"') or ""
    if b ~= nil then return b, reason end
  end
  return "oss-a", "router-unavailable"
end

function M.model_list()
  local out = {}
  for k, _ in pairs(M.PUBLIC_MODELS) do
    if k ~= "auto" then out[#out + 1] = k end
  end
  table.sort(out)
  return table.concat(out, ", ")
end

return M
