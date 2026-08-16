use super::shared::*;
use super::{
    emitter::SourceEmitter, IngestSource, NormalizedPartials, SourceMetadata, SourceRecordContext,
};
use regex::Regex;
use serde_json::Value;
use std::path::{Component, Path};
use std::sync::OnceLock;

pub(crate) static ANTIGRAVITY: Antigravity = Antigravity;

pub(crate) struct Antigravity;

fn uuid_pattern() -> &'static Regex {
    static UUID_RE: OnceLock<Regex> = OnceLock::new();
    UUID_RE.get_or_init(|| {
        Regex::new(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
            .expect("valid uuid regex")
    })
}

pub(crate) fn infer_antigravity_session_id(source_file: &str) -> String {
    let path = Path::new(source_file);
    for component in path.components() {
        if let Component::Normal(os_str) = component {
            if let Some(s) = os_str.to_str() {
                if uuid_pattern().is_match(s) {
                    return s.to_string();
                }
            }
        }
    }
    infer_session_id_from_file(source_file)
}

impl IngestSource for Antigravity {
    fn harness(&self) -> &'static str {
        "antigravity"
    }

    fn default_inference_provider(&self) -> Option<&'static str> {
        Some("google")
    }

    fn record_ts(&self, record: &Value) -> String {
        let created_at = to_str(record.get("created_at"));
        if !created_at.is_empty() {
            created_at
        } else {
            to_str(record.get("timestamp"))
        }
    }

    fn top_type(&self, record: &Value) -> String {
        to_str(record.get("type"))
    }

    fn source_metadata(&self, record: &Value) -> SourceMetadata {
        SourceMetadata {
            inference_provider: self
                .default_inference_provider()
                .unwrap_or_default()
                .to_string(),
            model_hint_fallback: to_str(record.get("model")),
        }
    }

    fn session_id(&self, record: &Value, ctx: &SourceRecordContext<'_>) -> String {
        let session_id = to_str(record.get("session_id"));
        if !session_id.trim().is_empty() {
            return session_id;
        }
        let conversation_id = to_str(record.get("conversation_id"));
        if !conversation_id.trim().is_empty() {
            return conversation_id;
        }
        if !ctx.session_hint.is_empty() {
            return ctx.session_hint.to_string();
        }
        infer_antigravity_session_id(ctx.source_file)
    }

    fn normalize(
        &self,
        record: &Value,
        ctx: &RecordContext<'_>,
        top_type: &str,
        base_uid: &str,
        model_hint: &str,
    ) -> NormalizedPartials {
        let mut emitter = SourceEmitter::new(ctx);
        normalize_antigravity_record(record, top_type, base_uid, model_hint, &mut emitter);
        emitter.finish()
    }
}

fn normalize_antigravity_record(
    record: &Value,
    top_type: &str,
    base_uid: &str,
    model_hint: &str,
    emitter: &mut SourceEmitter<'_>,
) {
    let step_index = to_u32(record.get("step_index"));
    let content = to_str(record.get("content"));
    let item_id = format!("step:{step_index}");
    let model = {
        let m = to_str(record.get("model"));
        if m.is_empty() {
            model_hint.to_string()
        } else {
            canonicalize_model("antigravity", &m)
        }
    };

    match top_type {
        "USER_INPUT" | "USER" => {
            let mut event = emitter
                .event_for_json(base_uid, "message", "message", "user", &content, record)
                .turn_index(step_index)
                .item_id(item_id)
                .content_types(["text"]);
            if !model.is_empty() {
                event = event.model(&model);
            }
            emitter.push_event(event);
        }
        "PLANNER_RESPONSE" => {
            let tool_calls = record
                .get("tool_calls")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or(&[]);

            let has_tool_calls = !tool_calls.is_empty();
            let mut content_types = Vec::new();
            if !content.is_empty() {
                content_types.push("text");
            }
            if has_tool_calls {
                content_types.push("tool_use");
            }
            if content_types.is_empty() {
                content_types.push("text");
            }

            let mut event = emitter
                .event_for_json(
                    base_uid,
                    "message",
                    "message",
                    "assistant",
                    &content,
                    record,
                )
                .turn_index(step_index)
                .item_id(item_id)
                .content_types(content_types);

            if !model.is_empty() {
                event = event.model(&model);
            }
            emitter.push_event(event);

            for (i, call) in tool_calls.iter().enumerate() {
                let tool_name = to_str(call.get("name"));
                let call_id = {
                    let id = to_str(call.get("id"));
                    if id.is_empty() {
                        let cid = to_str(call.get("call_id"));
                        if cid.is_empty() {
                            format!("step:{step_index}:call:{i}")
                        } else {
                            cid
                        }
                    } else {
                        id
                    }
                };
                let input_json = call
                    .get("args")
                    .map(compact_json)
                    .unwrap_or_else(|| "{}".to_string());
                emitter.push_tool_request(base_uid, &call_id, "", &tool_name, &input_json);
            }
        }
        "CHECKPOINT" | "CONVERSATION_HISTORY" | "SYSTEM" => {
            let event = emitter
                .event_for_json(
                    base_uid,
                    "session_meta",
                    "checkpoint",
                    "system",
                    &content,
                    record,
                )
                .turn_index(step_index)
                .item_id(item_id)
                .content_types(["text"]);
            emitter.push_event(event);
        }
        other => {
            // Check if this is a tool execution result (e.g. VIEW_FILE, RUN_COMMAND, LIST_DIR, etc.)
            let status = to_str(record.get("status"));
            let is_error = status.eq_ignore_ascii_case("ERROR");
            let tool_error = if is_error { 1 } else { 0 };
            let tool_name = other.to_ascii_lowercase();
            let call_id = {
                let id = to_str(record.get("call_id"));
                if id.is_empty() {
                    let cid = to_str(record.get("id"));
                    if cid.is_empty() {
                        format!("step:{step_index}")
                    } else {
                        cid
                    }
                } else {
                    id
                }
            };
            let output_json = compact_json(record);

            emitter.push_tool_response(
                base_uid,
                &call_id,
                "",
                &tool_name,
                tool_error,
                "",
                &output_json,
                &content,
            );

            let event = emitter
                .event_for_json(
                    base_uid,
                    "tool_result",
                    "tool_result",
                    "tool",
                    &content,
                    record,
                )
                .turn_index(step_index)
                .item_id(item_id)
                .tool_name(&tool_name)
                .tool_call_id(&call_id)
                .tool_phase("response")
                .tool_error(tool_error)
                .content_types(["tool_result"]);

            emitter.push_event(event);
        }
    }
}
