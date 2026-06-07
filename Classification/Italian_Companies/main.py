import pandas as pd 
import numpy as np 
import re

def main():
    # Load the data 
    df = pd.read_json("db_data.json")

    # Show the basic information 
    # Rows = records 
    # Columns = Info
    print(f"The shape of the dataset: {df.shape}\n")

    print(df.columns.tolist())


    print(f"\n First 5 rows: {df.head(5)}")

    # Missing values for each column 
    print(f"Missing values per column: {df.isna().sum()}")

    # Lets do some "Data Engineering"
  
    # Pattern we have for the addresses: " street - CAP - CITY (PROVINCE_CODE) "
    address_pattern = r"\s-\s*(?P<cap_from_address>\d{5})?\s*-\s*(?P<city_from_address>.*?)\s*\((?P<province_code_from_address>[A-Z]{2})?\)\s*$"

    # Extract parts from address (This applies the regex to every addresses in our dataset)
    extracted_address = df["address"].str.extract(address_pattern, flags=re.IGNORECASE)

    # Clean extracted values 
    # Use "str.strip() to remove extra spaces & str.upper() to change province code to uppercase"
    extracted_address["city_from_address"] = extracted_address["city_from_address"].str.strip()
    extracted_address["province_code_from_address"] = extracted_address["province_code_from_address"].str.upper()

    # Change empty strings into missing values as Pandas understand "NaN" as missing data 
    extracted_address = extracted_address.replace("", np.nan)

    # Add these extracted columns to the original dataframe we had 

    df = pd.concat([df, extracted_address], axis=1)



    # Check the results 
    df[[
        "address", 
        "cap_from_address",
        "city_from_address", 
        "province_code_from_address"
    ]].head()

    # As we can see, here cities are missed, while we have the addresses, so we can take advantage of the extraction 
    # and take the city from it.
    df[df["city"].isna() & df["address"].notna()][[
        "address",
        "city",
        "province",
        "region",
        "cap_from_address",
        "city_from_address",
        "province_code_from_address"
    ]].head(10)


    # Count how many non-missing values exist in each column.
    print("Rows with address:", df["address"].notna().sum())
    print("Rows where city was extracted:", df["city_from_address"].notna().sum())
    print("Rows where province code was extracted:", df["province_code_from_address"].notna().sum())
    print("Rows where CAP was extracted:", df["cap_from_address"].notna().sum())



    # Procvince_code to full_province_name mapping 

    province_code_to_name = {
        "AG": "Agrigento",
        "AL": "Alessandria",
        "AN": "Ancona",
        "AO": "Aosta",
        "AP": "Ascoli Piceno",
        "AQ": "L'Aquila",
        "AR": "Arezzo",
        "AT": "Asti",
        "AV": "Avellino",
        "BA": "Bari",
        "BG": "Bergamo",
        "BI": "Biella",
        "BL": "Belluno",
        "BN": "Benevento",
        "BO": "Bologna",
        "BR": "Brindisi",
        "BS": "Brescia",
        "BT": "Barletta-Andria-Trani",
        "BZ": "Bolzano",
        "CA": "Cagliari",
        "CB": "Campobasso",
        "CE": "Caserta",
        "CH": "Chieti",
        "CL": "Caltanissetta",
        "CN": "Cuneo",
        "CO": "Como",
        "CR": "Cremona",
        "CS": "Cosenza",
        "CT": "Catania",
        "CZ": "Catanzaro",
        "EN": "Enna",
        "FC": "Forlì-Cesena",
        "FE": "Ferrara",
        "FG": "Foggia",
        "FI": "Firenze",
        "FM": "Fermo",
        "FO": "Forlì-Cesena",   # legacy code, still found in old data
        "PS": "Pesaro e Urbino", # legacy code, still found in old data
        "FR": "Frosinone",
        "GE": "Genova",
        "GO": "Gorizia",
        "GR": "Grosseto",
        "IM": "Imperia",
        "IS": "Isernia",
        "KR": "Crotone",
        "LC": "Lecco",
        "LE": "Lecce",
        "LI": "Livorno",
        "LO": "Lodi",
        "LT": "Latina",
        "LU": "Lucca",
        "MB": "Monza e della Brianza",
        "MC": "Macerata",
        "ME": "Messina",
        "MI": "Milano",
        "MN": "Mantova",
        "MO": "Modena",
        "MS": "Massa Carrara",
        "MT": "Matera",
        "NA": "Napoli",
        "NO": "Novara",
        "NU": "Nuoro",
        "OR": "Oristano",
        "PA": "Palermo",
        "PC": "Piacenza",
        "PD": "Padova",
        "PE": "Pescara",
        "PG": "Perugia",
        "PI": "Pisa",
        "PN": "Pordenone",
        "PO": "Prato",
        "PR": "Parma",
        "PT": "Pistoia",
        "PU": "Pesaro e Urbino",
        "PV": "Pavia",
        "PZ": "Potenza",
        "RA": "Ravenna",
        "RC": "Reggio Calabria",
        "RE": "Reggio nell'Emilia",
        "RG": "Ragusa",
        "RI": "Rieti",
        "RM": "Roma",
        "RN": "Rimini",
        "RO": "Rovigo",
        "SA": "Salerno",
        "SI": "Siena",
        "SO": "Sondrio",
        "SP": "La Spezia",
        "SR": "Siracusa",
        "SS": "Sassari",
        "SU": "Sud Sardegna",
        "SV": "Savona",
        "TA": "Taranto",
        "TE": "Teramo",
        "TN": "Trento",
        "TO": "Torino",
        "TP": "Trapani",
        "TR": "Terni",
        "TS": "Trieste",
        "TV": "Treviso",
        "UD": "Udine",
        "VA": "Varese",
        "VB": "Verbano-Cusio-Ossola",
        "VC": "Vercelli",
        "VE": "Venezia",
        "VI": "Vicenza",
        "VR": "Verona",
        "VT": "Viterbo",
        "VV": "Vibo Valentia",
    }

    # full_province_name to Region_name mapping

    province_to_region = {
        # Abruzzo
        "Chieti": "Abruzzo",
        "L'Aquila": "Abruzzo",
        "Pescara": "Abruzzo",
        "Teramo": "Abruzzo",

        # Basilicata
        "Matera": "Basilicata",
        "Potenza": "Basilicata",

        # Calabria
        "Catanzaro": "Calabria",
        "Cosenza": "Calabria",
        "Crotone": "Calabria",
        "Reggio Calabria": "Calabria",
        "Vibo Valentia": "Calabria",

        # Campania
        "Avellino": "Campania",
        "Benevento": "Campania",
        "Caserta": "Campania",
        "Napoli": "Campania",
        "Salerno": "Campania",

        # Emilia-Romagna
        "Bologna": "Emilia-Romagna",
        "Ferrara": "Emilia-Romagna",
        "Forlì-Cesena": "Emilia-Romagna",
        "Modena": "Emilia-Romagna",
        "Parma": "Emilia-Romagna",
        "Piacenza": "Emilia-Romagna",
        "Ravenna": "Emilia-Romagna",
        "Reggio nell'Emilia": "Emilia-Romagna",
        "Rimini": "Emilia-Romagna",

        # Friuli-Venezia Giulia
        "Gorizia": "Friuli-Venezia Giulia",
        "Pordenone": "Friuli-Venezia Giulia",
        "Trieste": "Friuli-Venezia Giulia",
        "Udine": "Friuli-Venezia Giulia",

        # Lazio
        "Frosinone": "Lazio",
        "Latina": "Lazio",
        "Rieti": "Lazio",
        "Roma": "Lazio",
        "Viterbo": "Lazio",

        # Liguria
        "Genova": "Liguria",
        "Imperia": "Liguria",
        "La Spezia": "Liguria",
        "Savona": "Liguria",

        # Lombardia
        "Bergamo": "Lombardia",
        "Brescia": "Lombardia",
        "Como": "Lombardia",
        "Cremona": "Lombardia",
        "Lecco": "Lombardia",
        "Lodi": "Lombardia",
        "Mantova": "Lombardia",
        "Milano": "Lombardia",
        "Monza e della Brianza": "Lombardia",
        "Pavia": "Lombardia",
        "Sondrio": "Lombardia",
        "Varese": "Lombardia",

        # Marche
        "Ancona": "Marche",
        "Ascoli Piceno": "Marche",
        "Fermo": "Marche",
        "Macerata": "Marche",
        "Pesaro e Urbino": "Marche",

        # Molise
        "Campobasso": "Molise",
        "Isernia": "Molise",

        # Piemonte
        "Alessandria": "Piemonte",
        "Asti": "Piemonte",
        "Biella": "Piemonte",
        "Cuneo": "Piemonte",
        "Novara": "Piemonte",
        "Torino": "Piemonte",
        "Verbano-Cusio-Ossola": "Piemonte",
        "Vercelli": "Piemonte",

        # Puglia
        "Bari": "Puglia",
        "Barletta-Andria-Trani": "Puglia",
        "Brindisi": "Puglia",
        "Foggia": "Puglia",
        "Lecce": "Puglia",
        "Taranto": "Puglia",

        # Sardegna
        "Cagliari": "Sardegna",
        "Nuoro": "Sardegna",
        "Oristano": "Sardegna",
        "Sassari": "Sardegna",
        "Sud Sardegna": "Sardegna",

        # Sicilia
        "Agrigento": "Sicilia",
        "Caltanissetta": "Sicilia",
        "Catania": "Sicilia",
        "Enna": "Sicilia",
        "Messina": "Sicilia",
        "Palermo": "Sicilia",
        "Ragusa": "Sicilia",
        "Siracusa": "Sicilia",
        "Trapani": "Sicilia",

        # Toscana
        "Arezzo": "Toscana",
        "Firenze": "Toscana",
        "Grosseto": "Toscana",
        "Livorno": "Toscana",
        "Lucca": "Toscana",
        "Massa Carrara": "Toscana",
        "Pisa": "Toscana",
        "Pistoia": "Toscana",
        "Prato": "Toscana",
        "Siena": "Toscana",

        # Trentino-Alto Adige
        "Bolzano": "Trentino-Alto Adige",
        "Trento": "Trentino-Alto Adige",

        # Umbria
        "Perugia": "Umbria",
        "Terni": "Umbria",

        # Valle d'Aosta
        "Aosta": "Valle d'Aosta",

        # Veneto
        "Belluno": "Veneto",
        "Padova": "Veneto",
        "Rovigo": "Veneto",
        "Treviso": "Veneto",
        "Venezia": "Veneto",
        "Verona": "Veneto",
        "Vicenza": "Veneto",
    }


    # ### Lets test it out


    print(province_code_to_name["MI"])
    print(province_code_to_name["BA"])
    print(province_code_to_name["TO"])

    print("\n=======\n")

    print(province_to_region["Milano"])
    print(province_to_region["Bari"])
    print(province_to_region["Torino"])




    # A name normalization dictionary for common variants

    province_name_variants = {
        "milan": "Milano",
        "rome": "Roma",
        "naples": "Napoli",
        "florence": "Firenze",
        "turin": "Torino",
        "venice": "Venezia",
        "genoa": "Genova",
        "bologna": "Bologna",
        "palermo": "Palermo",
        "bari": "Bari",
    }   


    # Create clean columns for city, province, and region

    # This function will convert city names to title case and handle missing values 
    def title_case_city(value):
        if pd.isna(value):
            return np.nan
        return str(value).strip().title()

    # This function will standardize province names based on the extracted province code or original province name
    # value = original province name in the original DataFrame
    # province_code = province code extracted from address
    def clean_province(value, province_code=None):
        """
        Standardize province.
        Examples:
        Ba -> Bari
        BA -> Bari
        Bari -> Bari
        """

        # If the original province value exists, try to clean it first by removing extra spaces and checking if it's a 2-letter code
        if pd.notna(value):
            value = str(value).strip()

            # If province is written as a 2-letter code
            if len(value) == 2:
                code = value.upper()
                # Look up the full province name using the code
                return province_code_to_name.get(code, value)
            
            # Normalize English or variant spellings
            value_normalized = value.lower()
            if value_normalized in province_name_variants:
                return province_name_variants[value_normalized]

            # If province is already full name
            return value

        # If province is missing, use province code extracted from address
        if pd.notna(province_code):
            code = str(province_code).strip().upper()
            return province_code_to_name.get(code, np.nan)
        
    

        # If both original province and extracted province code are missing, return NaN
        return np.nan


    # Clean city:
    # If original city exists, use it.
    # Otherwise, use city extracted from address.
    df["city_clean"] = df["city"].combine_first(df["city_from_address"])
    # Apply title case to city names and handle missing values
    df["city_clean"] = df["city_clean"].apply(title_case_city)

    # Clean province
    # If original province exists, try to clean it. If not, use province code extracted from address to infer province name.
    df["province_clean"] = df.apply(
        lambda row: clean_province(
            row["province"],
            row["province_code_from_address"]
        ),
        axis=1
    )

    # Clean region
    # Use the cleaned province to infer region using the mapping. If province is missing,
    # we won't be able to infer region, so it will be NaN for now.
    df["region_clean"] = df["province_clean"].map(province_to_region)

    # If region could not be inferred, keep original region if it exists
    df["region_clean"] = df["region_clean"].combine_first(df["region"])


    # ### Test the result


    df[
        [
            "city", "city_from_address", "city_clean",
            "province", "province_code_from_address", "province_clean",
            "region", "region_clean"
        ]
    ].head(5)


    # ### Check cleaning results 


    changed_rows = df[
        (df["city"] != df["city_clean"]) |
        (df["province"] != df["province_clean"]) |
        (df["region"] != df["region_clean"])
    ]

    changed_rows[
        [
            "ragione_sociale",
            "address",
            "city", "city_clean",
            "province", "province_clean",
            "region", "region_clean"
        ]
    ].tail()


    # ### Compare the result before and after preprocessing 


    # Missing values before cleaning
    missing_before = df[["city", "province", "region"]].isna().sum()

    # Missing values after cleaning
    missing_after = df[["city_clean", "province_clean", "region_clean"]].isna().sum()

    # Rename indexes so they match
    missing_after.index = ["city", "province", "region"]

    # Compare before and after
    missing_comparison = pd.DataFrame({
        "missing_before": missing_before,
        "missing_after": missing_after,
        "fixed_values": missing_before - missing_after
    })

    print(missing_comparison)


    # ### Check the records which are still missed even after preprocessing


    uncertain_rows = df[
        (df["city_clean"].isna() & df["city_from_address"].notna()) |
        (df["province_clean"].isna() & df["province_code_from_address"].notna()) |
        (df["region_clean"].isna())
    ]
    uncertain_rows.to_csv("uncertain_records.csv", index=False)
    uncertain_rows.to_json("uncertain_records.json",  orient="records",indent=2,force_ascii=False)

    uncertain_rows[
        [
            "ragione_sociale",
            "address",
            "city", "city_from_address", "city_clean",
            "province", "province_code_from_address", "province_clean",
            "region", "region_clean"
        ]
    ].head()  



    uncertain_rows.shape

    # ### Creating the final clean output 

    cleaned_df = df.copy()

    # Replace original columns with cleaned columns
    cleaned_df["city"] = cleaned_df["city_clean"]
    cleaned_df["province"] = cleaned_df["province_clean"]
    cleaned_df["region"] = cleaned_df["region_clean"]

    # Keep only the required final columns
    cleaned_df = cleaned_df[
        [
            "ragione_sociale",
            "address",
            "city",
            "province",
            "region",
            "nation"
        ]
    ]

    # Show first rows of final cleaned data
    cleaned_df.head()


    # Save as JSON
    cleaned_df.to_json(
        "cleaned_companies.json",
        orient="records",
        indent=2,
        force_ascii=False
    )

    # Save as CSV
    cleaned_df.to_csv(
        "cleaned_companies.csv",
        index=False
    )


    # Final check of the cleaned data
    print("Final cleaned dataset shape:")
    print(cleaned_df.shape)

    print("\nMissing values in final cleaned data:")
    print(cleaned_df.isna().sum())


if __name__ == "__main__":
    main()