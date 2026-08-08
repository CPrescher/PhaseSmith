//! Small CIF 1.1 tokenizer and block/loop representation.

use std::collections::BTreeSet;

use super::{CifIoError, CifReadLimits};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TokenKind {
    Bare,
    Quoted,
    Text,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Token {
    value: String,
    kind: TokenKind,
    line: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum CifValue {
    Text(String),
    Unknown,
    Inapplicable,
}

impl CifValue {
    pub(super) fn text(&self) -> Option<&str> {
        match self {
            Self::Text(value) => Some(value),
            Self::Unknown | Self::Inapplicable => None,
        }
    }

    pub(super) fn missing_state(&self) -> Option<&'static str> {
        match self {
            Self::Unknown => Some("unknown"),
            Self::Inapplicable => Some("missing"),
            Self::Text(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
struct CifLoop {
    columns: Vec<(String, Vec<CifValue>)>,
}

#[derive(Clone, Debug)]
pub(super) struct CifBlock {
    pub(super) name: String,
    pairs: Vec<(String, CifValue)>,
    loops: Vec<CifLoop>,
}

impl CifBlock {
    pub(super) fn first_raw(&self, aliases: &[&str]) -> Option<(&str, &CifValue)> {
        aliases.iter().find_map(|alias| {
            let normalized = alias.to_ascii_lowercase();
            self.pairs
                .iter()
                .find(|(tag, _)| *tag == normalized)
                .map(|(tag, value)| (tag.as_str(), value))
                .or_else(|| {
                    self.loops.iter().find_map(|loop_| {
                        loop_
                            .columns
                            .iter()
                            .find(|(tag, _)| *tag == normalized)
                            .and_then(|(tag, values)| {
                                values.first().map(|value| (tag.as_str(), value))
                            })
                    })
                })
        })
    }

    pub(super) fn first_column(&self, aliases: &[&str]) -> Option<(&str, &[CifValue])> {
        aliases.iter().find_map(|alias| {
            let normalized = alias.to_ascii_lowercase();
            self.loops.iter().find_map(|loop_| {
                loop_
                    .columns
                    .iter()
                    .find(|(tag, _)| *tag == normalized)
                    .map(|(tag, values)| (tag.as_str(), values.as_slice()))
            })
        })
    }

    pub(super) fn all_tags(&self) -> BTreeSet<&str> {
        self.pairs
            .iter()
            .map(|(tag, _)| tag.as_str())
            .chain(
                self.loops
                    .iter()
                    .flat_map(|loop_| loop_.columns.iter().map(|(tag, _)| tag.as_str())),
            )
            .collect()
    }
}

#[derive(Clone, Debug)]
pub(super) struct CifDocument {
    pub(super) blocks: Vec<CifBlock>,
}

// CIF block parsing is intentionally linear and state-local. Keeping the loop
// grammar visible here makes control-token boundaries auditable.
#[allow(clippy::too_many_lines)]
pub(super) fn parse_document(text: &str, limits: CifReadLimits) -> Result<CifDocument, CifIoError> {
    let tokens = tokenize(text)?;
    let mut blocks = Vec::new();
    let mut index = 0;
    while index < tokens.len() {
        let heading = &tokens[index];
        if heading.kind != TokenKind::Bare
            || !heading.value.to_ascii_lowercase().starts_with("data_")
        {
            return Err(syntax_error(heading.line, "expected a data_ block heading"));
        }
        let raw_name = &heading.value[5..];
        let name = if raw_name.trim().is_empty() {
            format!("unnamed_{}", blocks.len() + 1)
        } else {
            raw_name.trim().to_owned()
        };
        if blocks.iter().any(|block: &CifBlock| block.name == name) {
            return Err(syntax_error(
                heading.line,
                "CIF data block names must be unique",
            ));
        }
        index += 1;
        let mut pairs = Vec::new();
        let mut loops = Vec::new();
        while index < tokens.len() && !is_data_heading(&tokens[index]) {
            let token = &tokens[index];
            if is_control(token, "loop_") {
                index += 1;
                let mut tags = Vec::new();
                while index < tokens.len() && is_tag(&tokens[index]) {
                    tags.push(tokens[index].value.to_ascii_lowercase());
                    index += 1;
                }
                if tags.is_empty() {
                    return Err(syntax_error(token.line, "loop_ has no tags"));
                }
                let value_start = index;
                while index < tokens.len() && !starts_item(&tokens[index]) {
                    index += 1;
                }
                let raw_values = &tokens[value_start..index];
                if raw_values.is_empty() || raw_values.len() % tags.len() != 0 {
                    return Err(syntax_error(
                        token.line,
                        "loop value count is not divisible by its column count",
                    ));
                }
                let tag_count = tags.len();
                let row_count = raw_values.len() / tag_count;
                if row_count > limits.max_loop_rows {
                    return Err(CifIoError::Limit {
                        message: "CIF loop exceeds max_loop_rows".to_owned(),
                    });
                }
                let columns = tags
                    .into_iter()
                    .enumerate()
                    .map(|(column, tag)| {
                        let values = raw_values
                            .iter()
                            .skip(column)
                            .step_by(tag_count)
                            .map(token_value)
                            .collect();
                        (tag, values)
                    })
                    .collect();
                loops.push(CifLoop { columns });
            } else if is_tag(token) {
                let tag = token.value.to_ascii_lowercase();
                index += 1;
                let value = tokens
                    .get(index)
                    .filter(|value| !starts_item(value))
                    .ok_or_else(|| syntax_error(token.line, "tag is missing its value"))?;
                pairs.push((tag, token_value(value)));
                index += 1;
            } else if is_control(token, "save_")
                || is_control(token, "stop_")
                || is_control(token, "global_")
            {
                return Err(syntax_error(
                    token.line,
                    "save frames, stop_, and global_ are not supported",
                ));
            } else {
                return Err(syntax_error(
                    token.line,
                    "unexpected value outside a tag or loop",
                ));
            }
        }
        blocks.push(CifBlock { name, pairs, loops });
        if blocks.len() > limits.max_blocks {
            return Err(CifIoError::Limit {
                message: "CIF document exceeds max_blocks".to_owned(),
            });
        }
    }
    if blocks.is_empty() {
        return Err(CifIoError::Import {
            message: "CIF document contains no data blocks".to_owned(),
        });
    }
    Ok(CifDocument { blocks })
}

// The tokenizer is one bounded state machine; extracting token branches would
// obscure the shared byte index and line accounting.
#[allow(clippy::too_many_lines)]
fn tokenize(text: &str) -> Result<Vec<Token>, CifIoError> {
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0;
    let mut line = 1;
    let mut line_start = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'\n' => {
                index += 1;
                line += 1;
                line_start = index;
            }
            b'\r' => {
                index += 1;
                if bytes.get(index) == Some(&b'\n') {
                    index += 1;
                }
                line += 1;
                line_start = index;
            }
            byte if byte.is_ascii_whitespace() => index += 1,
            b'#' => {
                while index < bytes.len() && !matches!(bytes[index], b'\n' | b'\r') {
                    index += 1;
                }
            }
            b';' if index == line_start => {
                let token_line = line;
                index += 1;
                let content_start = index;
                let mut closing = None;
                while index < bytes.len() {
                    if bytes[index] == b'\n' {
                        index += 1;
                        line += 1;
                        line_start = index;
                        if bytes.get(index) == Some(&b';') {
                            closing = Some(index);
                            break;
                        }
                    } else if bytes[index] == b'\r' {
                        index += 1;
                        if bytes.get(index) == Some(&b'\n') {
                            index += 1;
                        }
                        line += 1;
                        line_start = index;
                        if bytes.get(index) == Some(&b';') {
                            closing = Some(index);
                            break;
                        }
                    } else {
                        index += 1;
                    }
                }
                let closing = closing
                    .ok_or_else(|| syntax_error(token_line, "unterminated semicolon text field"))?;
                let value = text[content_start..closing].trim().to_owned();
                index = closing + 1;
                tokens.push(Token {
                    value,
                    kind: TokenKind::Text,
                    line: token_line,
                });
            }
            quote @ (b'\'' | b'"') => {
                let token_line = line;
                index += 1;
                let start = index;
                let mut closing = None;
                while index < bytes.len() {
                    if matches!(bytes[index], b'\n' | b'\r') {
                        return Err(syntax_error(token_line, "quoted value crosses a line"));
                    }
                    if bytes[index] == quote
                        && bytes.get(index + 1).is_none_or(u8::is_ascii_whitespace)
                    {
                        closing = Some(index);
                        break;
                    }
                    index += 1;
                }
                let closing =
                    closing.ok_or_else(|| syntax_error(token_line, "unterminated quoted value"))?;
                tokens.push(Token {
                    value: text[start..closing].to_owned(),
                    kind: TokenKind::Quoted,
                    line: token_line,
                });
                index = closing + 1;
            }
            byte if byte.is_ascii_control() => {
                return Err(syntax_error(line, "unsupported control character"));
            }
            _ => {
                let token_line = line;
                let start = index;
                while index < bytes.len()
                    && !bytes[index].is_ascii_whitespace()
                    && bytes[index] != b'#'
                {
                    index += 1;
                }
                tokens.push(Token {
                    value: text[start..index].to_owned(),
                    kind: TokenKind::Bare,
                    line: token_line,
                });
            }
        }
    }
    Ok(tokens)
}

fn token_value(token: &Token) -> CifValue {
    match (token.kind, token.value.as_str()) {
        (TokenKind::Bare, "?") => CifValue::Unknown,
        (TokenKind::Bare, ".") => CifValue::Inapplicable,
        _ => CifValue::Text(token.value.clone()),
    }
}

fn is_tag(token: &Token) -> bool {
    token.kind == TokenKind::Bare && token.value.starts_with('_')
}

fn is_data_heading(token: &Token) -> bool {
    token.kind == TokenKind::Bare && token.value.to_ascii_lowercase().starts_with("data_")
}

fn is_control(token: &Token, value: &str) -> bool {
    token.kind == TokenKind::Bare && token.value.eq_ignore_ascii_case(value)
}

fn starts_item(token: &Token) -> bool {
    is_tag(token)
        || is_data_heading(token)
        || ["loop_", "save_", "stop_", "global_"]
            .iter()
            .any(|value| is_control(token, value))
}

fn syntax_error(line: usize, message: impl Into<String>) -> CifIoError {
    CifIoError::Syntax {
        message: message.into(),
        line: Some(line),
    }
}
