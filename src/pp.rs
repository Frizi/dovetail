use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

#[derive(Debug)]
pub enum WgslError {
    ParserErr {
        error: String,
        line: usize,
        pos: usize,
    },
    IoErr(std::io::Error),
}

impl From<std::io::Error> for WgslError {
    fn from(err: std::io::Error) -> Self {
        Self::IoErr(err)
    }
}

fn advance(slice: &mut &str, pattern: &str) -> bool {
    *slice = slice.trim_start();
    if let Some(rest) = slice.strip_prefix(pattern) {
        *slice = rest;
        true
    } else {
        false
    }
}

fn include_syntax_error(line_number: usize, message: &str) -> WgslError {
    WgslError::ParserErr {
        error: message.into(),
        line: line_number,
        pos: 0,
    }
}

fn include_io_error(line_number: usize, err: std::io::Error) -> WgslError {
    WgslError::ParserErr {
        error: format!("{}", err),
        line: line_number,
        pos: 0,
    }
}

pub fn load_shader_preprocessed(path: &Path) -> Result<String, WgslError> {
    let mut set = HashSet::new();
    load_shader_preprocessed_recursive(path, &mut set)
}

fn load_shader_preprocessed_recursive(
    path: &Path,
    visited: &mut HashSet<PathBuf>,
) -> Result<String, WgslError> {
    let path = path.canonicalize()?;
    if visited.contains(&path) {
        return Ok(String::new());
    }
    visited.insert(path.clone());

    let source = std::fs::read_to_string(&path)?;
    let mut output = String::new();
    for (line_number, line) in source.lines().enumerate() {
        let mut tok = line.trim();
        if advance(&mut tok, "#include") {
            if !advance(&mut tok, "\"") {
                return Err(include_syntax_error(
                    line_number,
                    "expected '\"' after #include",
                ));
            }

            let (rel_path, rest) = tok.split_once("\"").ok_or_else(|| {
                include_syntax_error(line_number, "expected '\"' at the end of path")
            })?;
            tok = rest;

            if !(advance(&mut tok, ";") && tok.is_empty()) {
                return Err(include_syntax_error(
                    line_number,
                    "expected ';' after #include directive",
                ));
            }

            let full_path = path
                .parent()
                .map_or(PathBuf::from(rel_path), |parent| parent.join(rel_path));

            let included_source = load_shader_preprocessed_recursive(&full_path, visited).map_err(
                |err| match err {
                    WgslError::IoErr(e) => include_io_error(line_number + 1, e),
                    WgslError::ParserErr { error, pos, .. } => WgslError::ParserErr {
                        error,
                        line: line_number + 1,
                        pos,
                    },
                },
            )?;
            output.push_str(&included_source);
            output.push('\n');
        } else {
            output.push_str(line);
            output.push('\n');
        }
    }
    Ok(output)
}
