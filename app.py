from flask import Flask, render_template, request, send_file
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        tickers_input = request.form.get('tickers')
        if tickers_input:
            tickers = [t.strip().upper() for t in tickers_input.split(',')]
        else:
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

        # Now tickers is a list you can use directly
        print("Tickers list:", tickers)

        data = yf.download(tickers, start='2020-01-01', end='2024-12-31')['Close']
        data.dropna(axis=1, how='all', inplace=True) 
        returns = data.pct_change().dropna()
        mean_return = returns.mean(axis=0)
        cov_return = returns.cov()

        def portfolio_performance(weights, mean_returns, cov_matrix):
            returns = np.dot(weights, mean_returns) * 252  # annualized
            std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))  # annualized
            return returns, std_dev

        def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
            ret, std_dev = portfolio_performance(weights, mean_returns, cov_matrix)
            sharpe = (ret - risk_free_rate) / std_dev
            return -sharpe

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(tickers)))
        initial_guess = [1/len(tickers)] * len(tickers)

        opt_result = minimize(negative_sharpe, initial_guess, args=(mean_return, cov_return),
                              method='SLSQP', bounds=bounds, constraints=constraints)

        opt_weights = opt_result.x
        opt_ret, opt_vol = portfolio_performance(opt_weights, mean_return, cov_return)
        sharpe_ratio = (opt_ret - 0.01) / opt_vol

        def simulate_portfolios(num_portfolios):
            results = {'returns': [], 'volatility': [], 'sharpe': [], 'weights': []}
            for _ in range(num_portfolios):
                weights = np.random.dirichlet(np.ones(len(tickers)), size=1)[0]
                ret, vol = portfolio_performance(weights, mean_return, cov_return)
                sharpe = (ret - 0.01) / vol
                results['returns'].append(ret)
                results['volatility'].append(vol)
                results['sharpe'].append(sharpe)
                results['weights'].append(weights)
            return pd.DataFrame(results)

        portfolios = simulate_portfolios(5000)

        plt.figure(figsize=(10, 6))
        plt.scatter(portfolios['volatility'], portfolios['returns'], c=portfolios['sharpe'], cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.scatter(opt_vol, opt_ret, color='red', marker='*', s=200, label='Optimal Portfolio')
        plt.xlabel('Volatility (Risk)')
        plt.ylabel('Return')
        plt.title('Efficient Frontier')
        plt.legend()
        plt.grid(True)

        chart_path = "static/portfolio_prediction.png"
        plt.savefig(chart_path)
        plt.close()

        csv_file = "static/portfolio_data.csv"
        data.to_csv(csv_file)

        return render_template('index.html',
                               plt_chart=chart_path,
                               csv_file=csv_file,
                               tickers_used=data.columns.tolist())

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
