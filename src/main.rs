use dotenv::dotenv;
use tensara::{
    auth::{ensure_authenticated, ensure_authenticated_next},
    client, pretty,
    trpc::*,
    Parameters,
};

const COMPILED_MODAL_SLUG: &str = env!("COMPILED_MODAL_SLUG");

fn main() {
    #[cfg(debug_assertions)]
    dotenv().ok();
    let auth_info = ensure_authenticated();

    let username = &auth_info.github_username;
    let parameters: Parameters = Parameters::new(Some(username.clone()));

    let command_type = parameters.get_command_name();
    let dtype = parameters.get_dtype();
    let gpu_type = parameters.get_gpu_type();
    let problem_def = parameters.get_problem_def();
    let problem = parameters.get_problem();
    let language = parameters.get_language();

    let modal_slug =
        std::env::var("MODAL_SLUG").unwrap_or_else(|_| COMPILED_MODAL_SLUG.to_string());
    let endpoint = format!("{}/{}-{}", modal_slug, command_type, gpu_type);
    let endpoint = endpoint.as_str();

    let response = client::send_post_request(
        endpoint,
        &parameters.get_solution_code(),
        &problem,
        &problem_def,
        &dtype,
        &language,
    );

    match command_type.as_str() {
        "benchmark" => pretty::pretty_print_benchmark_response(response),
        "checker" => pretty::pretty_print_checker_streaming_response(response),
        "submit" => {
            if ensure_authenticated_next() {
                println!("Auth successfull....");
                let input = CreateSubmissionInput {
                        problemSlug: "vector-addition".to_string(),
                    code: "#include <cuda_runtime.h>\n\n__global__ void vector_add(const float* a, const float* b, float* c, size_t n) {\n    int i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < n) {\n        c[i] = a[i] + b[i];\n    }\n}\n\nextern \"C\" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {\n    int threads_per_block = 512;\n    int num_blocks = (n + threads_per_block - 1) / threads_per_block;\n    vector_add<<<num_blocks, threads_per_block>>>(d_input1, d_input2, d_output, n);\n}".to_string(),

                        language: "cuda".to_string(),
                        gpuType: "T4".to_string(),
                    };

                let submission = create_submission(
                    &auth_info,
                    input.problemSlug.as_str(),
                    input.code.as_str(),
                    input.language.as_str(),
                    input.gpuType.as_str(),
                )
                .unwrap();
                pretty::pretty_print_submit_streaming_response(direct_submit_read(
                    &auth_info,
                    submission.id.as_str(),
                ));
            } else {
                eprintln!("Authentication failed. Please paste your Next Token into your tensara auth file.");
            }
        }
        _ => unreachable!("Invalid command type"),
    }

    // Keep this code for debugging purposes, helps to see the raw response
    // let mut response_string = String::new();
    // response.read_to_string(&mut response_string).unwrap();
    // println!("{}", response_string);
}
