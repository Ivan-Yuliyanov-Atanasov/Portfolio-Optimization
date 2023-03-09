import matplotlib.pyplot as plt
import constants


def display(min_sd_port_sd, min_sd_port_return, max_sharpe_port_sd, tangency_port_sd, tangency_port_return, port_sd,
            port_returns, obj_sd, cml_exp_returns):
    plt.figure(figsize=(12, 6))
    # plt.plot(min_sd_port_sd, min_sd_port_return, 'r*', markersize=20.0, label="Min Variance Portfolio")
    # plt.plot(max_sharpe_port_sd, max_sharpe_port_return, 'b*', markersize=20.0, label="Max Sharpe Ratio Portfolio")
    plt.plot(tangency_port_sd, tangency_port_return, 'm*', markersize=20.0, label="Tangency Portfolio")
    plt.scatter(port_sd, port_returns, c=port_returns / port_sd)
    # plt.scatter(obj_sd, target_range, c=target_range / obj_sd)
    plt.plot(obj_sd[:85], constants.TARGET_RANGE[:85], label="Efficient Frontier")
    plt.plot(constants.CML_SD, cml_exp_returns, label="Capital Market Line")
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.title("Efficient Frontier and Capital Market Line")
    plt.legend()
    plt.show()
