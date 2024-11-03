import pandas as pd 

def retrieve_tickers():
    sectors = ["finance", "tech", "industrial", "energy"]
    sectors_tickers = {}

    # Get all tickers organized by sector in a dictionary 
    for i in range(len(sectors)):
        csv_data = pd.read_csv(f"/Users/andreybarriga/Documents/ML-DL/Quant-Finance/stock_data/{sectors[i]}-tick.csv")
        current_sector = sectors[i] # more readability 
        sectors_tickers.update({current_sector: csv_data["Index Holdings and weightings as of 6:38 PM ET 11/01/2024"]})
        sectors_tickers[current_sector] = sectors_tickers[current_sector].tolist()
        del sectors_tickers[current_sector][0] # deletes the "SYMBOL" that gets imported
        
    sectors_tickers["finance"][0] = "BRK-B" # small error of BRK.b instead of BRK-B

    return sectors_tickers



