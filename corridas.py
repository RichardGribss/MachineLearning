import pandas as pd

df = pd.read_csv("corridas.csv")

df.head()



# Seleção de variáveis
features = ["distancia_km", "tempo_min", "horario_pico", "chuva", "demanda"]
target = "preco"

X = df[features]
y = df[target]



# Separação treino e teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# Modelo inicial
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)



# Predição
y_pred = model.predict(X_test)



# Avaliação
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)

print(f"MAE do modelo inicial: {mae}")

# Modelo 1
from sklearn.ensemble import RandomForestRegressor

model1 = RandomForestRegressor(n_estimators=100, random_state=42)
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)

# Modelo 2
from sklearn.linear_model import LinearRegression

model2 = LinearRegression()
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)

# Avaliação
from sklearn.metrics import mean_absolute_error

mae1 = mean_absolute_error(y_test, pred1)
mae2 = mean_absolute_error(y_test, pred2)

print("Modelo 1 MAE:", mae1)
print("Modelo 2 MAE:", mae2)
# Criar novos exemplos (simulando novas corridas)

novas_corridas = pd.DataFrame({
    "distancia_km": [2, 5, 8, 3, 6, 10, 1.5, 7, 4, 9],
    "tempo_min": [6, 15, 22, 10, 18, 30, 5, 20, 12, 25],
    "horario_pico": [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    "chuva": [0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    "demanda": [0.4, 0.7, 0.9, 0.5, 0.8, 1.0, 0.3, 0.85, 0.6, 0.95]
})

# Fazer previsões com os dois modelos

pred_novas_1 = model1.predict(novas_corridas)
pred_novas_2 = model2.predict(novas_corridas)

print("Modelo 1 previsões:", pred_novas_1)
print("Modelo 2 previsões:", pred_novas_2)
