#!/bin/bash
# response_format json_schema and tool_choice "required" on /v1/chat/completions; then /health.
U=$1; K="$SGLANG_API_KEY"
echo "== json_schema"
curl -s -m 300 $U/v1/chat/completions -H "Authorization: Bearer $K" -H 'Content-Type: application/json' -d '{"model":"Nanbeige4.3-33B","messages":[{"role":"user","content":"Give me a city and its country."}],"max_tokens":200,"chat_template_kwargs":{"enable_thinking":false},"response_format":{"type":"json_schema","json_schema":{"name":"c","schema":{"type":"object","properties":{"city":{"type":"string"},"country":{"type":"string"}},"required":["city","country"]}}}}' | head -c 600; echo
echo "== tool_choice required"
curl -s -m 300 $U/v1/chat/completions -H "Authorization: Bearer $K" -H 'Content-Type: application/json' -d '{"model":"Nanbeige4.3-33B","messages":[{"role":"user","content":"Hi there"}],"max_tokens":300,"tool_choice":"required","tools":[{"type":"function","function":{"name":"get_weather","description":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}' | head -c 600; echo
sleep 3; echo "== health $(curl -s -o /dev/null -w '%{http_code}' $U/health)"
