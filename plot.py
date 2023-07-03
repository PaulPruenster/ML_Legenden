import matplotlib.pyplot as plt

with open('loss_CNN_50_epochen.txt', 'r') as file:
    data1 = [float(line.strip()) for line in file]

with open('loss_FC_50_epochen.txt', 'r') as file:
    data2 = [float(line.strip()) for line in file]

# Generate x-axis values (assuming equal spacing)
x = range(len(data1))

# Create a line graph
plt.plot(x, data1, label='CNN')
plt.plot(x, data2, label='Fully connected')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Cost functions')

plt.legend()

# Display the graph
plt.show()
