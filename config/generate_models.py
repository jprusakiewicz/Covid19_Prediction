import typer
import numpy as np
import yaml

app = typer.Typer()

@app.command()
def option_menu():
    options = {
        "1": "Keras",
        "2": "BaggingRegressor",
        "3": "RandomForestRegressor",
        "4": "DecisionTreeRegressor",
        "5": "KernelRidge",
        "6": "GaussianProcessRegressor",
        "7": "GradientBoostingRegressor",
        }

    typer.echo("Choose model type:")
    for key, value in options.items():
        typer.echo(f"{key}: {value}")

    selected_option = typer.prompt("Option: ")

    if selected_option in options:
        if selected_option == "1":
            typer.echo("Not implemented")
        if selected_option == "2":
            with open("config/test_configg.yaml", 'a') as config_file:
                selected_option = typer.prompt("n_estimators (type: start, stop, step):")
                selected_option = selected_option.split(',')
                for value in np.arange(float(selected_option[0]), float(selected_option[1]), float(selected_option[2])):
                    data = {'model': {'model_library': 'sklearn',  'type': "BaggingRegressor",
                            'params': {'random_state': 420,'n_jobs': -1, 'n_estimators': int(value.tolist())}}}
                    print(data)
                    yaml.dump(data, config_file, default_flow_style=False)
        if selected_option == "3":
            with open("config/test_configg.yaml", 'a') as config_file:
                selected_option = typer.prompt("n_estimators (type: start, stop, step):")
                selected_option = selected_option.split(',')
                for value in np.arange(float(selected_option[0]), float(selected_option[1]), float(selected_option[2])):
                    data = {'model': {'model_library': 'sklearn',  'type': "RandomForestRegressor",
                            'params': {'random_state': 420,'n_jobs': -1, 'n_estimators': int(value.tolist())}}}
                    print(data)
                    yaml.dump(data, config_file, default_flow_style=False)

        if selected_option == "4":
            with open("config/test_configg.yaml", 'a') as config_file:
                selected_option = typer.prompt("n_estimators (type: start, stop, step):")
                selected_option = selected_option.split(',')
                for value in np.arange(float(selected_option[0]), float(selected_option[1]), float(selected_option[2])):
                    data = {'model': {'model_library': 'sklearn',  'type': "DecisionTreeRegressor",
                            'params': {'random_state': 420,'n_jobs': -1, 'n_estimators': int(value.tolist())}}}
                    print(data)
                    yaml.dump(data, config_file, default_flow_style=False)
        if selected_option == "5":
            with open("config/test_configg.yaml", 'a') as config_file:
                selected_option = typer.prompt("alpha (type: start, stop, step):")
                selected_option = selected_option.split(',')
                for value in np.arange(float(selected_option[0]), float(selected_option[1]), float(selected_option[2])):
                    data = {'model': {'model_library': 'sklearn',  'type': "KernelRidge",
                            'params': {'alpha': int(value.tolist())}}}
                    print(data)
                    yaml.dump(data, config_file, default_flow_style=False)
        if selected_option == "6":
            with open("config/test_configg.yaml", 'a') as config_file:
                selected_option = typer.prompt("alpha (type: start, stop, step):")
                selected_option = selected_option.split(',')
                for value in np.arange(float(selected_option[0]), float(selected_option[1]), float(selected_option[2])):
                    data = {'model': {'model_library': 'sklearn',  'type': "GaussianProcessRegressor",
                            'params': {'alpha': int(value.tolist())}}}
                    print(data)
                    yaml.dump(data, config_file, default_flow_style=False)
        if selected_option == "7":
            with open("config/test_configg.yaml", 'a') as config_file:
                selected_option = typer.prompt("alpha (type: start, stop, step):")
                selected_option = selected_option.split(',')
                for value in np.arange(float(selected_option[0]), float(selected_option[1]), float(selected_option[2])):
                    data = {'model': {'model_library': 'sklearn',  'type': "GradientBoostingRegressor",
                            'params': {'alpha': int(value.tolist())}}}
                    print(data)
                    yaml.dump(data, config_file, default_flow_style=False)
    else:
        typer.echo("No such option!")

if __name__ == "__main__":
    app()
