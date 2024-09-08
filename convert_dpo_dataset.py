import json
import argparse

def reorganize_json(input_file, output_file):
    # Read the content of the input JSON file and convert it to the desired format
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            data = []
            for line in file:
                if line.strip():  # Skip empty lines
                    entry = json.loads(line.strip())  # Load each JSON object from the line

                    # Create a new structure in the desired format
                    new_entry = {
                        "conversations": [
                            {
                                "from": "human",
                                "value": entry["question"]  # Use the question as the human input
                            }
                        ],
                        "chosen": {
                            "from": "gpt",
                            "value": entry["response_chosen"]  # Use the chosen response as GPT output
                        },
                        "rejected": {
                            "from": "gpt",
                            "value": entry["response_rejected"]  # Use the rejected response as GPT rejected output
                        }
                    }

                    data.append(new_entry)

        # Save the newly structured JSON data to the output file
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=2)

        print(f"Reorganized JSON has been saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' does not exist.")
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON in the file '{input_file}'.")
    except UnicodeDecodeError as e:
        print(f"Error: Failed to read the file '{input_file}' due to encoding issues: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize JSON content from input to output file.")
    parser.add_argument("--in_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--out_file", type=str, required=True, help="Path to the output JSON file")

    args = parser.parse_args()

    # Call the function with provided arguments
    reorganize_json(args.in_file, args.out_file)
