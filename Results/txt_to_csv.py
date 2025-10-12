import pandas as pd


filenames = [
    "PBE-D3BJ.txt",
    "NN_PBE.txt",
    "XAlpha.txt",
    "NN_XAlpha.txt",
    "NN_PBE_star.txt",
]

def extract_energies_from_txt(filename):
    functional = filename.rstrip(".txt")
    error_dict = {
        "System":[],
        f"{functional} Error":[],
    }
    with open(filename, 'r') as file:
        lines = [line.strip().split() for line in file.readlines()[:50]]
    for line in lines:
        if len(line)==4:
            system = line[0]
            error_dict["System"].append(system)
            error_dict[f"{functional} Error"].append(float(line[-1]))
    df = pd.DataFrame(error_dict)
    df.set_index('System', inplace=True)
    return df


def visualize(df):
    pass


def main():
    results = []
    for filename in filenames:
        df = extract_energies_from_txt(filename)
        results.append(df)
    df_total = pd.concat(results, axis=1)
    database = df_total.index.to_series().apply(lambda x: "-".join(x.split('-')[:-1]))
    df_total['Database'] = database

    df_total.to_csv("DietGMTKN55_results.csv")

if __name__ == "__main__":
    main()