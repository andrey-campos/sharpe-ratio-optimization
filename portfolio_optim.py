import numpy as np
import pandas as pd
from scipy.optimize import minimize 
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from decimal import Decimal, getcontext
from typing import Tuple


"""
Maximizing the sharpe ratio to perform portfolio optimization:
    - Use tickers from sector variable, that has 10 stocks per sector
    - Grab data from yfinance and preprocess the closing stock's price
    - Diversify these stocks and choose stocks that have the lowest average
    correlation
    - Calculate daily historical returns of each stock and each sector
    - Use historical returns to then calculate the portfolio's expected return 
        expected returns using exponentially weighted moving averages (EWMA)
    - Calculate variance and volatility of the portfolio
    - Use bounds method to intialize weights and also use bounds variable for optimization
    - Use sharpe ratio as a metric of risk adjusted return and to be used as the objective
    - Maximize the sharpe ratio using scipy's minimize
    - Compare metrics of regular vs optimized portfolio
    - Visualize the portfolio's most important aspects
"""

# 10 random possible stocks per sector
sectors = {
    "finance": ["BRK-B", "JPM", "BAC", "GS", "V", "MA", "BX", "MS", "PYPL", "BLK"],
    "tech": ["NVDA", "MSFT", "AAPL", "ORCL", "ADBE", "UBER", "GOOG", "META", "CSCO", "ZM"],
    "industrial": ["RTX", "GE", "CAT", "LMT", "UPS", "UNP", "BA", "HON", "GD", "ADP"],
    "energy": ["XOM", "TXGE", "ET", "FANG", "OXY", "BKR", "MPC", "CQP", "HAL", "LNG"],
}

class Portfolio():
    def __init__(self, num_stocks: int, start_date: str, end_date: str, 
        annual_rate: float, trading_days: int, one_minus_lambda: float,
        upper_bound: float, lower_bound: float) -> dict:
        self.portfolio: dict = {
            "Finance": pd.DataFrame(),
            "Tech": pd.DataFrame(),
            "Industrial": pd.DataFrame(),
            "Energy": pd.DataFrame()
        }
        # important variables that will be used throughout program
        self.start_date = start_date
        self.end_date = end_date
        self.annual_rate = annual_rate
        self.trading_days = trading_days
        self.num_stocks = num_stocks
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.portfolio_expected_return: float = 0.0
        self.expected_returns_vector: list[float] = []
        self.one_minus_lambda = one_minus_lambda
        # initial stock weight as (1, 20) vector with equal intialization
        self.stock_weight: np.array =  np.full((1, 20), (1 / (len(self.portfolio) * self.num_stocks)))



    # pre-process stock data
    def prepare_data(self, sectors: dict, log_data=True) -> dict:   
        # Create a new list containing each stock, where stocks comes from sectors.values(), and stock comes from each stocks list
        stocks = [stock for stocks in sectors.values() for stock in stocks]
        try:
            stock_data = yf.download(tickers=stocks, start=self.start_date, end=self.end_date, auto_adjust=True)["Close"]

            if (len(stocks) - len(stock_data.columns)) != 0:
                print(f"Missing stock data inside data")
                return stock_data 
            
            else:
                print(f"All stock data downloaded succesfully")
    
                sector_mapping = {
                    "finance": "Finance",
                    "tech": "Tech",
                    "industrial": "Industrial",
                    "energy": "Energy"
                }
                
                # log all returns for better statistical properties
                for sector, stocks in sectors.items():
                    portfolio_sector = sector_mapping[sector]
                    sector_data = stock_data[stocks]

                    # Choose between logged or unlogged data: default -> True
                    if log_data:
                        self.portfolio[portfolio_sector] = np.log(sector_data)
                    else:
                        self.portfolio[portfolio_sector] = sector_data

                    # filling any NaNs or Nas with values that come before it
                    self.portfolio[portfolio_sector] = self.portfolio[portfolio_sector].ffill()
                return self.portfolio
    
        except Exception as error: 
            print(f"Recieved {error} while retrieving historical stock data, please try again.")

    # Diversify stocks by grabbing stocks with only the lowest correlation amongst each other
    def diversify_sectors(self, show_lowest_corr=False) -> dict:
        getcontext().prec = 10 # for exact calculations
        assets = self.prepare_data(sectors, log_data=True)
        diversified_assets = {}

        # Transfer all diversified assets that we're found to a diversified portfolio with yf data
        diversified_portfolio = assets.copy()

        # make a correlation matrix per sector to use it to calculate average stock correlation
        for sector, data in assets.items():
            corr_matrix = data.corr()
            # use dictionary to save stock ticker and avg correlation
            avg_correlations = {}

            # outerloop allows us to loop through all row stock names per sector
            for stock in corr_matrix.index:
                correlations = [Decimal(str(x)) for other_stock, x in corr_matrix[stock].items() if other_stock != stock]
                avg_corr = sum(correlations) / Decimal(len(correlations))

                avg_correlations[stock] = float(avg_corr)

            # make all dictionary key value pairs to inside sectors to series's
            diversified_assets[sector] = pd.Series(avg_correlations)
            diversified_assets[sector] = diversified_assets[sector].nsmallest(self.num_stocks, "first")

        # Make a modified portfolio that removes the assets that AREN'T diversified:
        for dp_sector, dp_data in diversified_portfolio.items():
            dataframe_stocks, series_stocks = [], []
            dataframe_stocks = [col for col in diversified_portfolio[dp_sector].columns]
            for index in diversified_assets[dp_sector].index:
                series_stocks.append(index)
            # filters all stocks that are not inside the diversifed stocks variable:
            diversified_portfolio[dp_sector] = diversified_portfolio[dp_sector].drop([stock for stock in dataframe_stocks if stock not in series_stocks], axis=1)
        
        # Allows to see the average correlation for the stocks that were picked
        if show_lowest_corr:
            print(f"\n\nDiversified portfolio stocks lowest correlation averages: \n{diversified_assets} \n\nType of portfolio: {type(diversified_assets)} \nType of data inside sectors keys: {type(diversified_assets["Finance"])}")
        return diversified_portfolio
    
    # inputted portfolio will provide the data needed to calculate all historical returns
    def historical_returns(self, input_portfolio: dict) -> dict:
        # prevents actual altercations to data by copying portfolio param
        hr_portfolio = input_portfolio.copy()

        # use pandas pct change to get period over period change
        for sector, data in hr_portfolio.items():
            # Remove oldest day to get rid of all NaN's & Replace inf's , NaNs, or Na's with zeroes:
            hr_portfolio[sector] = (hr_portfolio[sector].pct_change()).drop([self.start_date])
            hr_portfolio[sector] = hr_portfolio[sector].replace([np.inf, -np.inf], 0)

        return hr_portfolio
        
    # Make's new datastructure that stores the expected return for every stock, and stock weight:
    def expected_returns(self, historical_returns: dict, show_port_exp_return=False) -> dict:
        # Avoid any actual modification of data
        hrs = historical_returns.copy()
        analyzed_portfolio = self.portfolio.copy()
        weight_index = 0 

        # flatten just incase its (4, 5), num of sector's * num of stocks in sectors , 1 / num of sectors, and stock tickers to save all organized stock tickers
        self.stock_weight, self.sector_weight, self.stock_tickers = self.stock_weight.flatten(), 1/len(hrs), []
       
        for sector, data in hrs.items():
            # Reset list to allow us to add correct stock names per sector
            stock_names = []
            # turn this dictionary into a new pd dataframe eventually
            analyzed_portfolio[sector] = {
                    "Expected_Returns": [],
                    "Stock_Percantage": [],
                }
            
            for series_name, series in data.items(): 
                # Calculate stock's expected return using pandas ewm method and mean to get EWMA, get last element to represent EWMA
                stock_expected_return = (series.ewm(alpha=self.one_minus_lambda).mean())[-1]
        
                # use this vector that contains each individual stock expected return for the optimization
                self.expected_returns_vector.append(stock_expected_return)
                # while other for actual use and metrics
                stock_names.append(series_name)
                self.stock_tickers.append(series_name)

                 # append most recent expected return from EWMA algorthim & the stock weight
                analyzed_portfolio[sector]["Stock_Percantage"].append(self.stock_weight[weight_index])
                weight_index += 1
                analyzed_portfolio[sector]["Expected_Returns"].append(stock_expected_return)


            # when all data is finished being added, turn it to a dataframe and have indexes be tickers
            analyzed_portfolio[sector] = pd.DataFrame(analyzed_portfolio[sector], index=stock_names)


        # look for expected return's column, then use this column calculate each column's expected return by iteravely adding to it per sector
        for sector, data in analyzed_portfolio.items():
            # turn both columns in to numpy arrays to allow to get dot product for sector's expected return
            expected_returns = np.array(data["Expected_Returns"])
            sector_weight = np.array(data["Stock_Percantage"])
            sector_expected_return = np.dot(sector_weight, expected_returns)

            # incremently add it so get a portfolio expected return 
            self.portfolio_expected_return += sector_expected_return.item()

        # if user wants to see the portfolio's actual expected return; return the analyzed portfolio & portfolio expected return
        if show_port_exp_return:
            print(f"\nThe portfolio's entire expected return based on historical return's: {self.portfolio_expected_return}\n")
            return analyzed_portfolio, self.portfolio_expected_return
        
        return analyzed_portfolio
    
    # Returns covariance of all historical returns; needs diversified portfolio as input not the historical return portfolio
    def variance(self, input_portfolio: dict) -> Tuple[np.array, np.array]:
        try:
            hr_portfolio = self.historical_returns(input_portfolio)
            # returns all dfs in historical return portfolio as a list with each element as a df, inits stocks weights vector
            hr_dataframes = [hr_portfolio[keys] for keys in hr_portfolio.keys()]
            # reshape to be able to calculate variance
            self.stock_weight = self.stock_weight.reshape(1, 20)
            # excludes one df to be able to use the 'join' method to merge rext of dfs
            portfolio_variance = (hr_dataframes[0].join(hr_dataframes[1:])).cov()
            portfolio_variance = (np.dot(self.stock_weight, portfolio_variance)) @ self.stock_weight.T
            # square root to return both variance and volatility
            return portfolio_variance, np.sqrt(portfolio_variance)
        
        except KeyError as error:
            print(f"\nNeed to input normal/diversified portfolio! This explains: {error} missing date error!\nmethod 'variance' will calculate historical returns for you")

    
    # input weights as a pandas series turned into list; will return tuple of bounds and list of weights
    def bounds(self, returns_portfolio: dict) -> Tuple[tuple, list[float]]:
        # turn all sectors dataframes into huge dataframe to get all stock weights
        return_dataframe = pd.concat([returns_portfolio[keys] for keys in returns_portfolio.keys()])
        weights = return_dataframe["Stock_Percantage"].to_list()
        # use tuple comprehension to make bounds for scipy minimize, also choose maximum value for weight
        bounds = tuple((self.lower_bound, self.upper_bound) for weight in range(len(weights)))
        return bounds, weights

    # calculate the annual sharpe ratio by annualizing return and volatility
    def sharpe_ratio(self, weights: list[float], portfolio_vol: float,
        negative_sharpe=False) -> float:
       # Calculate portfolio return based on weights
        portfolio_return = np.dot(weights, self.expected_returns_vector)
        
        # Annualize return and volatility to match annual rate and get a annaul sharpe ratio
        annualized_return = portfolio_return * self.trading_days
        annualized_vol = portfolio_vol * np.sqrt(self.trading_days)
        
        # Calculate Sharpe ratio
        sharpe = (annualized_return - self.annual_rate) / annualized_vol
        
        # Make it negative for minimization if needed
        if negative_sharpe: 
            sharpe = -sharpe
        return sharpe.item() 

    # maximize sharpe ratio; with scipy minimize, bounds & set constraints; same arguments as sharpe ratio
    def maximize_sharpe(self, weights:list[float], 
        bounds:tuple,  portfolio_vol: float) -> object:
            optimized_sharpe = minimize(
            fun=self.sharpe_ratio,
            x0=weights,
            args=(portfolio_vol, True),
            bounds=bounds,
            method="SLSQP",
            # No short selling; all weights summed must equal 1; at solution 
            constraints=[
                    {"type": "eq", "fun":  lambda weight: np.sum(weight) - 1},
                    {"type": "ineq", "fun": lambda weight: weight}
                ]
            )
            return optimized_sharpe
    

    # new datastructure of optimized portfolio will basically be expected return portfolio with additional optimized weights column
    def optimized_portfolio(self, maxed_sharpe: object, hrs: dict, 
        diversified_portfolio: dict) -> Tuple[dict, tuple]:
        # copy of portfolios and optimized weights of maximized sharpe ratio to prevent altercations; weights are organized
        optimized_port, historical_return_port, diversified_port = self.portfolio.copy(),  hrs.copy(), diversified_portfolio.copy()

        # transformed to a 2d array with 4 rows and 5 columns, allows for each row to be added iteravely per sector
        self.stock_weight: np.array = ((maxed_sharpe.x).copy()).reshape(len(optimized_port), self.num_stocks)

        # use all previous methods to handle the creation of the new data structure
        optimized_port, optim_expected_return = self.expected_returns(historical_returns=historical_return_port, show_port_exp_return=True)
        optim_variance, optim_volatility = self.variance(input_portfolio=diversified_port)
        # return the optimized portfolio, its expected return, and the variance and volatility of the portfolio
        return optimized_port, optim_expected_return, optim_variance, optim_volatility

    # user gets to decide whether he wants to see one sector, or whole portfolio asset allocation;
    def asset_allocation(self, show_one_sector=False, input_portfolio=None, sector=None) -> plt:
        if show_one_sector:
            sector_tickers = [index for index in input_portfolio[sector].index]
            sector_weights = (input_portfolio[sector]["Stock_Percantage"]).tolist()
            fig, ax = plt.subplots()
            ax.set_title(f"Asset allocation for {sector}:", fontsize=15)
            ax.pie(sector_weights, labels=sector_tickers)
            plt.show()
        else:
            # returns double list so we index to get simplze list with stock tickers
            tickers, percantages = self.stock_tickers, (self.stock_weight.tolist())[0]
            fig, ax = plt.subplots()
            ax.set_title("Asset allocation for all stock tickers", fontsize=15)
            ax.pie(percantages, labels=tickers)
            plt.show()
    
    # show correlation between a portfolio's stocks:
    def correlation_heatmap(self, input_portfolio: dict) -> plt:
        for sector, data in input_portfolio.items():
            sns.heatmap(data.corr(numeric_only=True), cmap="YlGnBu", annot=True)
            plt.title(f"{sector}'s correlation heatmap:", fontsize=15)
            plt.show()



# important instance variables that will be used throughtout program
portfolio = Portfolio(num_stocks=5, start_date="2022-8-8", end_date="2023-8-12", 
    annual_rate=0.0327, trading_days=254, lower_bound=0.0, upper_bound=0.10,one_minus_lambda=0.04) 

# Shows assets data; returned as logs 
assets = portfolio.prepare_data(sectors, log_data=True)
print(f"Logged returns and preprocessed data : {assets}\n\n")

# Shows diversified portfolio assets, flexibity for how many stocks we pick, allows for checking average correlations:
diversified_portfolio = portfolio.diversify_sectors(show_lowest_corr=False)
print(f"\n\nComplete and diversified stocks: \n{diversified_portfolio}")

# allows you to choose which portfolio to calculate historical returns
historical_returns_portfolio = portfolio.historical_returns(diversified_portfolio)
print(f"\nHistorical returns for each stock inside each sector (per week basis):\n{historical_returns_portfolio}")

# provides the expected return & annual return for each stock as a new data structure with rows as stocks and cols as data
return_portfolio, portfolio_expected_return = portfolio.expected_returns(historical_returns_portfolio, show_port_exp_return=True)
print(f"\nExpected returns of portfolio: {return_portfolio}\n\nExpceted return of portfolio saved in a variable: {portfolio_expected_return}")

# Will calculate the variance & volatility for the historical return data; depends on if inputted normal or diversified portfolio
portfolio_var, portfolio_volatility = portfolio.variance(diversified_portfolio)
print(f"\nVariance of portfolio that was chosen for historical returns: {portfolio_var.item()}\nPortfolio Volatility (std): {portfolio_volatility.item()}")

# returns bounds for optimization and the list of weights from return portfolio
bounds, weights = portfolio.bounds(return_portfolio)
print(f"\nthe bounds for optimization:\n{bounds, len(bounds)}\nThe list of weights/stock percantage per stock from return portfolio:\n{weights, len(weights)}\n\n")

# Calculates the sharpe ratio per portfolio expected return, the annual rate, the voltality and amount of trading days
sharpe_ratio = portfolio.sharpe_ratio(weights, portfolio_volatility, negative_sharpe=True)
print(f"The negative (arg neg sharpe set True) of the sharpe ratio of the portfolio: {sharpe_ratio}\n")

# returns the optimized sharpe ratio and uses the negative sharpe ratio and list of weights / stock percantages:
optimized_sharpe = portfolio.maximize_sharpe(weights, bounds, portfolio_volatility)
print(f"\nOptimized sharpe ratio and weights for the portfolio:\n{optimized_sharpe}")

# Finally: compare the metrics of the unoptimized versus optimized portfolio: 
optimized_portfolio, optimzed_return, optimized_variance, optimized_volatility = portfolio.optimized_portfolio(optimized_sharpe, historical_returns_portfolio, diversified_portfolio)
print(f"\nThe new optimized portfolio and it's metrics:\n{optimized_portfolio}\nThe new total expected return {optimzed_return}\nThe new variance and volatility: {optimized_variance.item(), optimized_volatility.item()}")
print(f"\nOld expected return portfolio: {return_portfolio}\nOld total expected return: {portfolio_expected_return}\nOld variance and volatility: {portfolio_var.item(), portfolio_volatility.item()}")
print(f"\n\nFinal metric: old sharpe ratio: {sharpe_ratio}\nNew sharpe ratio: {abs(optimized_sharpe.fun)}")

# More visual aspects to compare metrics of unoptimized versus optimized portfolio:
portfolio.asset_allocation(show_one_sector=True, input_portfolio=optimized_portfolio, sector="Tech")
portfolio.correlation_heatmap(assets)