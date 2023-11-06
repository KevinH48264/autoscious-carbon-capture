# Using LLM to generate topics
# Improvement: add rate limiting error handling so you don't hardcode the wait time. To catch errors, use these catch exceptions from BabyAGI: https://github.com/yoheinakajima/babyagi/blob/main/babyagi.py

# imports
import dotenv
import os
import openai
import json
import time

dotenv.load_dotenv()
# openai.api_key = os.getenv("OPENAI_GPT4_API_KEY")
openai.api_key = os.getenv("OPENAI_CORP_API_KEY")

# models
EMBEDDING_MODEL = "text-embedding-ada-003"
GPT_MODEL = "gpt-3.5-turbo"
# GPT_MODEL = "gpt-4"

# for bulk openai message, no stream
def chat_openai_high_temp(prompt="Tell me to ask you a prompt", model=GPT_MODEL, chat_history=[], temperature=0):
    # define message conversation for model
    if chat_history:
        messages = chat_history
    else:
        messages = [
            {"role": "system", "content": "You are a helpful and educated carbon capture research consultant and an educated and helpful researcher and programmer. Answer as correctly, clearly, and concisely as possible. You give first-rate answers."},
        ]
    messages.append({"role": "user", "content": prompt})

    # create the chat completion
    print("Prompt: ", prompt)
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=1.0,
    )
    print("Completion info: ", completion)
    text_answer = completion["choices"][0]["message"]["content"]

    # updated conversation history
    messages.append({"role": "assistant", "content": text_answer})

    return text_answer, messages

def chat_openai(prompt="Tell me to ask you a prompt", model=GPT_MODEL, chat_history=[], temperature=0, verbose=False, system_prompt="You are a helpful assistant. Answer as correctly, clearly, and concisely as possible.", functions=[], function_call="auto", available_functions={}):
    # To deal with rate limits, just wait

    # define message conversation for model
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    if chat_history:
        messages += chat_history
    messages.append({"role": "user", "content": prompt})

    count = 0
    while True:
        if count > 2:
            print("Retry happened 3 times! Kicking in to save you tokens.")
            break

        try:
            count += 1
            # create the chat completion
            if verbose:
                print("Prompt messages: ", messages)
            
            if functions:
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    functions=functions,
                    function_call=function_call,
                )
            else:
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
            if verbose:
                print("Completion info: ", completion)
            response_message = completion["choices"][0]["message"]

            # Handle function calls if there are any
            count = 0
            while "function_call" in response_message and response_message.get("function_call") and count < 3:
                if verbose: 
                    print("Entering function call: ", response_message.get("function_call"))
                # Step 3: call the function
                # Note: the JSON response may not always be valid; be sure to handle errors
                function_name = response_message["function_call"]["name"]
                function_to_call = available_functions[function_name]
                function_args = json.loads(response_message["function_call"]["arguments"])
                function_response = function_to_call(**function_args)

                # Step 4: send the info on the function call and function response to GPT
                # TODO: potentially fix this because there's some funky conversation here.
                # Commenting out below to potentially address this
                # messages.append(response_message)  # extend conversation with assistant's reply

                if verbose:
                    print("Function call output: ", function_response) # Note: This returns the answer and the full reasoning in skill agent

                # TODO: You need to let the agent know what function was called.
                # print("\n\nresponse_message", response_message, "\n\nfunction_name", function_name, "\n\nfunction_args", function_args, "\n\nfunction_response", function_response)
                messages.append(
                    {
                        "role": "function",
                        "name": function_name,
                        "content": f"Response from running function {function_name} with arguments {function_args}: {function_response}"
                    }
                )  # extend conversation with function response

                if verbose:
                    print("Messages now: ", messages)
                response_message = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0613",
                    messages=messages,
                    temperature=temperature,
                    functions=functions,
                    function_call=function_call,
                )["choices"][0]["message"]  # get a new response from GPT where it can see the function response

                # Repeat if there's still a function call
                count += 1

            if verbose:
                print("Function calling response: ", response_message)

            messages.append(response_message) # So you can review the full chat history

            return response_message, messages
        except openai.error.RateLimitError:
            print(
                "   *** The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.Timeout:
            print(
                "   *** OpenAI API timeout occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIError:
            print(
                "   *** OpenAI API error occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIConnectionError:
            print(
                "   *** OpenAI API connection error occurred. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.InvalidRequestError:
            print(
                "   *** OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.ServiceUnavailableError:
            print(
                "   *** OpenAI API service unavailable. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break