import pandas as pd


def convert_encoding(input_file, output_file, from_encoding, to_encoding='utf-8'):
    # Read the file with the specified encoding
    df = pd.read_csv(input_file, encoding=from_encoding)

    # Save the file with the new encoding
    df.to_csv(output_file, index=False, encoding=to_encoding)


# Example usage
convert_encoding('Data/Temp.csv', 'Data/Temp_copy.csv', 'ISO-8859-1')
