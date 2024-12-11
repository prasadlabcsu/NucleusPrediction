# Importing packages

import matplotlib.pyplot as plt

# Reading the history file

h_file = open("C:\\Users\\sebas\\OneDrive\\Desktop\\HeLa_Images_TIFF\\HeLa UNET Files\\History_Group_Ble_Model_4.txt")
h_text = h_file.read()

# Finding the number of batches

batches = '0'
counter = 0
cursor = h_text[112 + counter]
while (cursor != '}'):
    batches += cursor
    counter += 1
    cursor = h_text[112 + counter]
    
n_batches = int(batches)
n_batches = 194
# Reading the training data per batch

train_history = {'batch': [], 'rec': []}

e = 0
for k in range(len(h_text)):
    cursor = h_text[k]
    if cursor == 'E':
        e_n = ''
        counter = k
        while (cursor != '{'):
            counter += 1
            cursor = h_text[counter]
        counter += 1
        cursor = h_text[counter]
        while (cursor != '}'):
            e_n += cursor
            counter += 1
            cursor = h_text[counter]
        e = int(e_n)
    if cursor == 'B':
        batch_n = ''
        counter = k
        while (cursor != '{'):
            counter += 1
            cursor = h_text[counter]
        counter += 1
        cursor = h_text[counter]
        while (cursor != '}'):
            batch_n += cursor
            counter += 1
            cursor = h_text[counter]
        train_history['batch'].append(int(batch_n) + n_batches*(e - 1))
        rec_loss = ''
        while (cursor != 'R'):
            counter += 1
            cursor = h_text[counter]
        while (cursor != ':'):
            counter += 1
            cursor = h_text[counter]
        counter += 1
        cursor = h_text[counter]
        while (cursor != '\n'):
            rec_loss += cursor
            counter += 1
            cursor = h_text[counter]
        train_history['rec'].append(float(rec_loss))

# Rescaling the loss values

#rec_max = 75*512*512
#train_history['rec'] = [x/rec_max for x in train_history['rec']]

# Plotting training data by batch

plt.title('History of UNET Model Training Losses')
plt.plot(train_history['batch'], train_history['rec'], 'g')
plt.grid(visible=True, axis='both')
plt.xlabel('Iterations')
plt.ylabel('Reconstruction Loss')
plt.xlim([1, max(train_history['batch'])])
plt.ylim([0, max(train_history['rec'])])
plt.show()

# Reading validation data per batch

valid_history = {'epoch': [], 'rec': []}

e = 0
for k in range(len(h_text)):
    cursor = h_text[k]
    if cursor == 'V':
        h_check = h_text[k+18]
        if h_check == 'h':
            epoch_n = ''
            counter = k
            while (cursor != 'h'):
                counter += 1
                cursor = h_text[counter]
            counter += 1
            cursor = h_text[counter]
            if cursor == 'a':
                pass
            else:
                while (cursor != ':'):
                    epoch_n += cursor
                    counter += 1
                    cursor = h_text[counter]
                valid_history['epoch'].append(int(epoch_n))
                rec_loss = ''
                while (cursor != 'R'):
                    counter += 1
                    cursor = h_text[counter]
                while (cursor != ':'):
                    counter += 1
                    cursor = h_text[counter]
                counter += 1
                cursor = h_text[counter]
                while (cursor != '\n'):
                    rec_loss += cursor
                    counter += 1
                    cursor = h_text[counter]
                valid_history['rec'].append(float(rec_loss))

# Rescaling the loss values

#rec_max = max(valid_history['rec'])
#valid_history['rec'] = [x/rec_max for x in valid_history['rec']]

# Plotting training data by batch

plt.title('History of Model Validation Losses')
plt.plot(valid_history['epoch'], valid_history['rec'], 'g')
plt.grid(visible=True, axis='both')
plt.xlabel('Epochs')
plt.ylabel('Reconstruction Loss')
plt.xlim([1, max(valid_history['epoch'])])
plt.ylim([0, max(valid_history['rec'])])
plt.show()
