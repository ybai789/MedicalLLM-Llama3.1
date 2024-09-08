import json
import argparse

def reorganize_json(input_file, output_file):
    # Read the content of the JSON file line by line with UTF-8 encoding
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            data = []
            for line in file:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))  # Append each JSON object

        # Create a new structure with exactly the same content but in the desired format
        new_structure = [{"instruction": entry["instruction"], "input": entry["input"], "output": entry["output"]} for entry in data]

        # Convert to JSON format for output
        formatted_output = json.dumps(new_structure, indent=2)

        # Save the new structure to the output file with UTF-8 encoding
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write(formatted_output)

        print(f"Reorganized JSON has been saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' does not exist.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON in the file '{input_file}'.")
    except UnicodeDecodeError as e:
        print(f"Error: Failed to read file '{input_file}' due to encoding issues: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize JSON content from input to output file.")
    parser.add_argument("--in_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--out_file", type=str, required=True, help="Path to the output JSON file")

    args = parser.parse_args()

    # Call the function with provided arguments
    reorganize_json(args.in_file, args.out_file)
