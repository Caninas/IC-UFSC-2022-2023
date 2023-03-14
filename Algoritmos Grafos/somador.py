soma = 0
l = 0

while True:
    i = input("")

    if i == "c":
        break

    soma += int(i)
    l += 1

print(soma / l)