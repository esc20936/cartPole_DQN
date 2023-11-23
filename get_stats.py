# open the file in read mode
f = open("train_log.txt", "r")
# read the file
content = f.read()


# split the file into lines
lines = content.split("\n")
lines = lines[:-1]
scores = []
indexs = []
i = 1

for line in lines:
    # split the line into words
    _ = line.split(',')
    print(_)
    score_string = _[1]

    # remove spaces from the score string
    score_string = score_string.replace(" ", "")

    # get score
    score = score_string.split(":")[1]

    # add score to scores list
    scores.append(int(score))
    indexs.append(i)
    i += 1

from matplotlib import pyplot as plt

# plot the scores
plt.plot(indexs, scores)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Training Progress")

# set y axis ticks to integers from 0 to max score
max_score = max(scores)
plt.yticks(range(0, max_score,50))



plt.show()
