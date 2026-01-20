
x = df.drop(columns=['diagnosis'])
y = df['diagnosis']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)
