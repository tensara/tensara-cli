use crate::trpc::ProblemParameter;
/*
* Generate starter code from ProblemParameter
*/
pub fn generate_starter_code(
    parameters: &[ProblemParameter],
    language: &str,
    data_type: &str,
) -> String {
    let cpp_types = |dtype: &str| match dtype {
        "float" => "float",
        "int" => "int",
        "double" => "double",
        _ => "float",
    };

    let python_types = |dtype: &str| match dtype {
        "float" => "float",
        "int" => "int",
        "double" => "float", // Triton doesn't support double
        _ => "float",
    };

    if language == "cuda" {
        let names: Vec<_> = parameters
            .iter()
            .filter(|p| p.pointer.as_deref() == Some("true"))
            .map(|p| p.name.clone())
            .collect();

        let param_str = parameters
            .iter()
            .map(|p| {
                let mut type_str = if p.ty == "[VAR]" {
                    cpp_types(data_type).to_string()
                } else {
                    p.ty.clone()
                };

                if p.const_.as_deref() == Some("true") {
                    type_str = format!("const {}", type_str);
                }

                if p.pointer.as_deref() == Some("true") {
                    type_str = format!("{}*", type_str);
                }

                format!("{} {}", type_str, p.name)
            })
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "#include <cuda_runtime.h>

// Note: {} are all device pointers to {} arrays
extern \"C\" void solution({}) {{    
}}",
            names.join(", "),
            data_type,
            param_str
        )
    } else if language == "python" {
        let names: Vec<_> = parameters
            .iter()
            .filter(|p| p.pointer.as_deref() == Some("true"))
            .map(|p| p.name.clone())
            .collect();

        let param_str = parameters
            .iter()
            .map(|p| {
                if p.pointer.as_deref() == Some("true") {
                    p.name.clone()
                } else if p.ty == "[VAR]" {
                    format!("{}: {}", p.name, python_types(data_type))
                } else {
                    format!("{}: int", p.name)
                }
            })
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "import triton\nimport triton.language as tl

# Note: {} are all {} device tensors
def solution({}):
    pass",
            names.join(", "),
            data_type,
            param_str
        )
    } else {
        "".to_string()
    }
}

pub fn validate_code(code: &str, language: &str) -> Result<(), String> {
    if language == "python" {
        if code.contains("torch.") || code.contains("import torch") {
            return Err("You cannot use PyTorch in the code!".into());
        }

        let exec_regex = regex::Regex::new(r"exec\s*\(\s*[^)]*\)").unwrap();
        if exec_regex.is_match(code) {
            return Err("You cannot use exec() in the code!".into());
        }
    }

    Ok(())
}

/*
* TODO Create sol.cu with input string in the given directory
*/
