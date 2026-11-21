-- Organizer-only eval listener (bound to 127.0.0.1). Exposes the two OSS
-- models plus Opus through one OpenAI-compatible endpoint so offline
-- benchmark harnesses can run every model through the same code path.
-- Nothing on this listener is shadowed or recorded.
local C = dofile("/etc/envoy/common.lua")

local EVAL_MODELS = { [C.OSS_A] = "oss-a", [C.OSS_B] = "oss-b", [C.SHADOW_MODEL] = "anthropic" }

function envoy_on_request(h)
  C.strip_internal_headers(h)
  if h:headers():get(":method") ~= "POST" then return end
  local body = h:body()
  if body == nil or body:length() == 0 then return end
  local raw = body:getBytes(0, body:length())
  local model = C.extract_model(raw, EVAL_MODELS)
  if model == nil then
    C.json_error(h, 400, "unknown model; available: " .. C.OSS_A .. ", " .. C.OSS_B .. ", " .. C.SHADOW_MODEL)
    return
  end
  h:headers():add("x-bench-backend", EVAL_MODELS[model])
  h:clearRouteCache()
end
