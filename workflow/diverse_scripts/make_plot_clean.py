import matplotlib.pyplot as plt

# Dados originais
classes_originais = ['1.x.x.x', '2.x.x.x', '3.x.x.x', '4.x.x.x', '5.x.x.x', '6.x.x.x']
quantidades_originais = [19, 26, 37, 10, 6, 1]

# Subclassificação
subclasse = '3.1.x.x'
quantidade_subclasse = 12

# Cálculo do restante da classe
quantidade_restante = quantidades_originais[2] - quantidade_subclasse

# Criar nova lista combinando tudo
novas_classes = ['1.x.x.x', '2.x.x.x', '3.1.x.x', '3.x.x.x', '4.x.x.x', '5.x.x.x', '6.x.x.x']
novas_quantidades = [
    quantidades_originais[0],
    quantidades_originais[1],
    quantidade_subclasse,
    quantidade_restante,
    quantidades_originais[3],
    quantidades_originais[4],
    quantidades_originais[5]
]

# Cores personalizadas (opcional)
colors = ['blue', 'cyan','lightgreen', 'green', 'orange', 'red', 'purple']

# Gráfico de pizza
plt.figure(figsize=(8, 8))
plt.pie(novas_quantidades, labels=novas_classes,colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of EC numbers')
plt.legend()
plt.tight_layout()
plt.show()

