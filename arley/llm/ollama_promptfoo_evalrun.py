# https://www.promptfoo.dev/docs/usage/self-hosting

# docker build --build-arg NEXT_PUBLIC_PROMPTFOO_BASE_URL=http://localhost:3000 -t promptfoo-ui .
# git clone https://github.com/promptfoo/promptfoo.git
# cd promptfoo

# {{ messages | dump }}

# mkdir -p ${HOME}/promptfoo/DATA
# docker run -d --name promptfoo_container -p 3000:3000 -v ${HOME}/promptfoo/DATA:/root/.promptfoo arleyelasticcio/promptfoo:latest

# share-run
#-e PROMPTFOO_SHARE_STORE_TYPE=filesystem -e PROMPTFOO_REMOTE_API_BASE_URL=http://localhost:3000 -e PROMPTFOO_REMOTE_APP_BASE_URL=http://localhost:3000 promptfoo share -y


# These configuration options can also be set under the sharing property of your promptfoo config:
#
# sharing:
#   apiBaseUrl: http://localhost:3000
#   appBaseUrl: http://localhost:3000


# run promptfoo-ui:
# docker run -it --rm --name promptfoo-ui --add-host=host.docker.internal:host-gateway --add-host=ollama.intra.fara.de:10.6.0.10 -e OLLAMA_BASE_URL=http://ollama.intra.fara.de --network=host -e OPENAI_API_KEY=BLARGH -e PROMPTFOO_REMOTE_API_BASE_URL=http://127.0.0.1:3000 -e PROMPTFOO_REMOTE_APP_BASE_URL=http://127.0.0.1:3000 -p 3000:3000 -v ${HOME}/promptfoo/DATA:/root/.promptfoo -v ${HOME}/promptfoo/DATA_CLI:/app/PF arleyelasticcio/promptfoo:latest

# init etc.
# docker exec -it promptfoo-ui /usr/local/bin/promptfoo /usr/local/bin/promptfoo init PF



# OLD:
# docker run -it --rm --name promptfoo-cli -e PROMPTFOO_SHARE_STORE_TYPE=filesystem -e PROMPTFOO_REMOTE_API_BASE_URL=http://localhost:3000 -e PROMPTFOO_REMOTE_APP_BASE_URL=http://localhost:3000 -v ${HOME}/promptfoo/DATA_CLI:/app/PF arleyelasticcio/promptfoo:latest /usr/local/bin/promptfoo init PF




# promptfoo-cli
# evalrun
# docker run -it --rm --name promptfoo-cli --network=host --add-host=ollama.intra.fara.de:10.6.0.10 -e OLLAMA_BASE_URL=http://ollama.intra.fara.de -e PROMPTFOO_REMOTE_API_BASE_URL=http://127.0.0.1:3000 -e PROMPTFOO_REMOTE_APP_BASE_URL=http://127.0.0.1:3000 -v ${HOME}/promptfoo/DATA_CLI:/app/PF -v ${HOME}/promptfoo/DATA:/root/.promptfoo arleyelasticcio/promptfoo:latest /usr/local/bin/promptfoo eval -c /app/PF/promptfooconfig.yaml --interactive-providers --max-concurrency=1 --no-cache
# docker exec -it promptfoo-ui -e  PROMPTFOO_REMOTE_API_BASE_URL=http://127.0.0.1:3000 -e PROMPTFOO_REMOTE_APP_BASE_URL=http://127.0.0.1:3000 /usr/local/bin/promptfoo eval -o /app/PF/output.json -c /app/PF/promptfooconfig.yaml



# NO-CACHE:
# docker exec -it promptfoo-ui /usr/local/bin/promptfoo eval -o /app/PF/output.json -c /app/PF/promptfooconfig.yaml --interactive-providers --no-cache

# --no-write
# OLD:
# docker run -it --rm --name promptfoo-cli --add-host=ollama.intra.fara.de:10.6.0.10 -e OLLAMA_BASE_URL=http://ollama.intra.fara.de -e PROMPTFOO_SHARE_STORE_TYPE=filesystem -e PROMPTFOO_REMOTE_API_BASE_URL=http://localhost:3000 -e PROMPTFOO_REMOTE_APP_BASE_URL=http://localhost:3000 -v ${HOME}/promptfoo/DATA_CLI:/app/PF arleyelasticcio/promptfoo:latest /usr/local/bin/promptfoo eval -o /app/PF/output.json -c /app/PF/promptfooconfig.yaml




import json
import sys


# Python variables
# For Python, the approach is similar. Define a Python script that includes a get_var function to generate your variable's value. The function should accept var_name, prompt, and other_vars.
#
# tests:
#   - vars:
#       context: file://fetch_dynamic_context.py
#
# fetch_dynamic_context.py:
#
# def get_var(var_name, prompt, other_vars):
#     # Example logic to dynamically generate variable content
#     if var_name == 'context':
#         return {
#             'output': f"Context for {other_vars['input']} in prompt: {prompt}"
#         }
#     return {'output': 'default context'}
#
#     # Handle potential errors
#     # return { 'error': 'Error message' }

def my_prompt_function(context: dict) -> str:

    provider: dict = context['providers']
    provider_id: str = provider['id']  # ex. openai:gpt-4o or bedrock:anthropic.claude-3-sonnet-20240229-v1:0
    provider_label: str | None = provider.get('label') # exists if set in promptfoo config.

    variables: dict = context['vars'] # access the test case variables

    return (
        f"Describe {variables['topic']} concisely, comparing it to the Python"
        " programming language."
    )

if __name__ == "__main__":
    # If you don't specify a `function_name` in the provider string, it will run the main
    print(my_prompt_function(json.loads(sys.argv[1])))



# Python script
# Your Python script should implement a function that accepts a prompt, options, and context as arguments. It should return a JSON-encoded ProviderResponse.
#
# The ProviderResponse must include an output field containing the result of the API call.
# Optionally, it can include an error field if something goes wrong, and a tokenUsage field to report the number of tokens used.
# By default, supported functions are call_api, call_embedding_api, and call_classification_api. To override the function name, specify the script like so: python:my_script.py:function_name
# Here's an example of a Python script that could be used with the Python provider, which includes handling for the prompt, options, and context:
#
# # my_script.py
# import json
#
# def call_api(prompt, options, context):
#     # The 'options' parameter contains additional configuration for the API call.
#     config = options.get('config', None)
#     additional_option = config.get('additionalOption', None)
#
#     # The 'context' parameter provides info about which vars were used to create the final prompt.
#     user_variable = context['vars'].get('userVariable', None)
#
#     # The prompt is the final prompt string after the variables have been processed.
#     # Custom logic to process the prompt goes here.
#     # For instance, you might call an external API or run some computations.
#     output = call_llm(prompt)
#
#
#     # The result should be a dictionary with at least an 'output' field.
#     result = {
#         "output": output,
#     }
#
#     if some_error_condition:
#         result['error'] = "An error occurred during processing"
#
#     if token_usage_calculated:
#         # If you want to report token usage, you can set the 'tokenUsage' field.
#         result['tokenUsage'] = {"total": token_count, "prompt": prompt_token_count, "completion": completion_token_count}
#
#     return result
#
# def call_embedding_api(prompt):
#     # Returns ProviderEmbeddingResponse
#     pass
#
# def call_classification_api(prompt):
#     # Returns ProviderClassificationResponse
#     pass
#
#
# Types
# The types passed into the Python script function and the ProviderResponse return type are defined as follows:
#
# class ProviderOptions:
#     id: Optional[str]
#     config: Optional[Dict[str, Any]]
#
# class CallApiContextParams:
#     vars: Dict[str, str]
#
# class TokenUsage:
#     total: int
#     prompt: int
#     completion: int
#
# class ProviderResponse:
#     output: Optional[Union[str, Dict[str, Any]]]
#     error: Optional[str]
#     tokenUsage: Optional[TokenUsage]
#     cost: Optional[float]
#     cached: Optional[bool]
#     logProbs: Optional[List[float]]
#
# class ProviderEmbeddingResponse:
#     embedding: List[float]
#     tokenUsage: Optional[TokenUsage]
#     cached: Optional[bool]
#
# class ProviderClassificationResponse:
#     classification: Dict[str, Any]
#     tokenUsage: Optional[TokenUsage]
#     cached: Optional[bool]
#
#
# Setting the Python executable
# In some scenarios, you may need to specify a custom Python executable. This is particularly useful when working with virtual environments or when the default Python path does not point to the desired Python interpreter.
#
# Here's an example of how you can override the Python executable using the pythonExecutable option:
#
# providers:
#   - id: 'python:my_script.py'
#     config:
#       pythonExecutable: /path/to/python3.11