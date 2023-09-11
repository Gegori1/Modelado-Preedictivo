import openai
from os import environ, system, getenv
import pyperclip
import sys
from time import sleep
import pprint


openai.api_key = getenv('AO_API_KEY_TWO')

# ==================================================================================================
# Useful functions
# ==================================================================================================

def get_completion_gpt(messages: dict) -> str:  
    """
    Get the completion of a prompt using the specified model.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.6, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def cls():
    """
    Clear the terminal.
    """
    system('clear')

# ==================================================================================================
# Runable code
# ==================================================================================================


if __name__ == "__main__":

    name = sys.argv
    if len(name) > 1:
        custom = name[1]
        print("Custom chat: " + custom)

        from custom_chat import messages_dict

        if custom in messages_dict.keys():
            messages = messages_dict[custom]
        
        else:
            print("No custom chat found with that name.")
            print("Available custom chats: " + str(list(messages_dict.keys())))
            x = input("do you want to use one of the standard chats? (y/n): ")
            if x.lower() in ["y", "yes", "ye"]:
                print("Available standard chats: " + str(list(messages_dict.keys())))
                custom = input("Enter the name of the standard chat you want to use: ")
                if custom in messages_dict.keys():
                    messages = messages_dict[custom]
                else:
                    print("No standard chat found with that name.")
                    print("Available standard chats: " + str(list(messages_dict.keys())))
                    sleep(5)
                    exit()
        
    else:
        system_behav = input("Enter the system behavior: ")

        messages = [{"role": "system", "content": system_behav}]

    while True:
        user_input = input("\033[93m" + "You: " + "\033[0m")

        if user_input == "exit":
            break

        if user_input == "clear":
            cls()
            continue

        # change system behavior
        if user_input == "\x04":
            if messages[0]['role'] == "system":
                system_behav = input("Enter the new system behavior: ")
                messages[0] = {"role": "system", "content": system_behav}
                continue
            else:
                x = input("There is no system behavior to change. Do you want to add one? (y/n): ")
                if x.lower() in ["y", "yes", "ye"]:
                    system_behav = input("Enter the new system behavior: ")
                    messages.insert(0, {"role": "system", "content": system_behav})
                    continue
                continue
        
        # clear conversation
        if user_input == "clear conversation":
            x = input("Are you sure you want to clear the conversation? (y/n): ")
            if x.lower() in ["y", "yes", "ye"]:
                messages = [messages[0]]
                print("Conversation cleared.")
            x1 = input("Do you want to add a new system behavior? (y/n): ")
            if x1.lower() in ["y", "yes", "ye"]:
                messages = []
                system_behav = input("Enter the new system behavior: ")
                messages.append({"role": "system", "content": system_behav})
                continue
            continue

        # to clipboard
        if user_input == "clip":
            pyperclip.copy(messages[-1]["content"])
            print("Copied to clipboard.")
            continue

        # remove last entry
        if user_input == "undo":
            if len(messages) > 2:
                messages.pop()
            continue

        # add example
        if user_input == "add example":
            x = input("Enter the user input: ")
            messages.insert(1, {"role": "user", "content": x})
            y = input("Enter the system output: ")
            messages.insert(2, {"role": "assistant", "content": y})
            continue

        # print help
        if user_input == "help":
            print("Commands:")
            print("clear - clear the terminal")
            print("clear conversation - clear the conversation")
            print("clip - copy the assistant's last response to the clipboard")
            print("undo - undo the last user input")
            print("add example - add an example to the conversation")
            print("exit - exit the program")
            print("help - show this help message")
            continue

        if user_input == "print messages":
            pprint.pprint(messages)
            continue

        if user_input == "\x01":
            print("Press Ctrl+Z (Windows) to exit:")
            print("\033[93m" + "You: " + "\033[0m", end="", flush=True)
            user_input = sys.stdin.read()
            

        # append new user message to messages
        messages.append({"role": "user", "content": user_input})

        # get the assistant's response
        try:
            out = get_completion_gpt(messages)
            messages.append({"role": "assistant", "content": out})
        except openai.error.APIConnectionError:
            print("Error comunicationg with OpenAi. Check you connection")
            continue

        print("\033[93m" + "System: " + "\033[0m" + out)


        