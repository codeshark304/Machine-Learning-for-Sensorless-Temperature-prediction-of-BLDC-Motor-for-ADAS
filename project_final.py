import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import customtkinter as ctk
from tkinter import messagebox

warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\harik\Documents\Machine Learning\temperature_data_access.csv")
df = df.drop(['speed', 'current', 'power_loss', 'speed_mean', 'current_mean', 'speed_std', 'current_std'], axis=1)

df_small = df.sample(frac=0.1, random_state=42)

features = ['time', 'temp_in', 'temp_out']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_small[features])
X = scaled_data[:, :-1]
y = scaled_data[:, 2]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

lr = LinearRegression()
lr.fit(x_train, y_train)
ypred_lr = lr.predict(x_test)
mse_linear = mean_squared_error(y_test, ypred_lr)
r2_linear = r2_score(y_test, ypred_lr)
print("Multiple Linear Regression:")
print(f"Mean Squared Error: {mse_linear:.4f}")
print(f"R^2 Score: {r2_linear:.4f}\n")

svr = SVR(kernel='linear', verbose=False)
svr.fit(x_train, y_train)
ypred_svr = svr.predict(x_test)
mse_svr = mean_squared_error(y_test, ypred_svr)
r2_svr = r2_score(y_test, ypred_svr)
print("Support Vector Regressor:")
print(f"Mean Squared Error: {mse_svr:.4f}")
print(f"R^2 Score: {r2_svr:.4f}\n")
    
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
ypred_rf = rf.predict(x_test)
mse_rf = mean_squared_error(y_test, ypred_rf)
r2_rf = r2_score(y_test, ypred_rf)
print("Random Forest Regressor:")
print(f"Mean Squared Error: {mse_rf:.4f}")
print(f"R^2 Score: {r2_rf:.4f}\n")

xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(x_train, y_train)
ypred_xgb = xgb.predict(x_test)
mse_xgb = mean_squared_error(y_test, ypred_xgb)
r2_xgb = r2_score(y_test, ypred_xgb)
print("XGBoost Regressor:")
print(f"Mean Squared Error: {mse_xgb:.4f}")
print(f"R^2 Score: {r2_xgb:.4f}\n")

results = pd.DataFrame({
    'Model': ['Multiple Linear Regression', 'Support Vector Regressor', 'Random Forest Regressor', 'XGBoost Regressor'],
    'Mean Squared Error': [mse_linear, mse_svr, mse_rf, mse_xgb],
    'R^2 Score': [r2_linear, r2_svr, r2_rf, r2_xgb]
})
print(results)

def plot_predictions(y_test, y_pred, model_name):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name}: True vs Predicted')
    plt.show()

plot_predictions(y_test, ypred_lr, 'Multiple Linear Regression')
plot_predictions(y_test, ypred_svr, 'Support Vector Regressor')
plot_predictions(y_test, ypred_rf, 'Random Forest Regressor')
plot_predictions(y_test, ypred_xgb, 'XGBoost Regressor')

def predict_temperature():
    try:
        temp_in = float(entry_temp_in.get())
        time = float(entry_time.get())
        if temp_in < 20 or time < 30:
            messagebox.showerror("Input Error", "Temperature must be at least 20 degrees and time must be at least 30 minutes.")
            return
        user_data = scaler.transform([[time, temp_in, 0]])
        scaled_features = user_data[:, :-1]
        predicted_temp_out = rf.predict(scaled_features)
        original_scale = scaler.inverse_transform([[time, temp_in, predicted_temp_out[0]]])
        temp_out = original_scale[0, 2]
        result_text = f"Predicted Output Temperature: {temp_out:.2f} degrees Celsius"
        if temp_out > 29.25275:
            result_text += "\nActivate Cooling systems"
        else:
            result_text += "\nCooling systems not required"
        messagebox.showinfo("Prediction Result", result_text)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for temperature and time.")

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Temperature Prediction System")
root.geometry("400x300")

label_temp_in = ctk.CTkLabel(root, text="Input Temperature (minimum 20 degrees):")
label_temp_in.pack(pady=10)
entry_temp_in = ctk.CTkEntry(root)
entry_temp_in.pack(pady=5)

label_time = ctk.CTkLabel(root, text="Time (minimum 30 minutes):")
label_time.pack(pady=10)
entry_time = ctk.CTkEntry(root)
entry_time.pack(pady=5)

predict_button = ctk.CTkButton(root, text="Predict Output Temperature", command=predict_temperature)
predict_button.pack(pady=20)

root.mainloop()
