########################## Hyperparameters ##########################
FRAME_LENGTH=256
FRAME_STEP=160
FFT_LENGTH=384

REMOVE_SILENCE=False

BATCH_SIZE=1
EPOCHS=200

#CNN1
FILTERS1=32
KERNEL_SIZE1=[11,41]
STRIDES1=[2,2]
PADDING1="same"

#CNN2
FILTERS2=32
KERNEL_SIZE2=[11,21]
STRIDES2=[1,2]
PADDING2="same"

#RNN
RNN_LAYERS=0
RNN_UNITS=512

#Search Type
IS_GREEDY=False
BEAM_WIDTH=100
TOP_PATHS=1



#%%
##########################      Code        ##########################
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display
from jiwer import wer
from os import listdir
import visualkeras

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
    
    
data_path = r'C:\Users\Malaz_N\Desktop'
wavs_path = data_path + '\\waves\\'
metadata_path = data_path + '\\quran.csv'


metadata_df = pd.read_csv(metadata_path)
metadata_df =metadata_df[metadata_df['Surah']>=78]
metadata_df=metadata_df[["File name", "Text"]]
metadata_df.columns = ["File name", "Text"]
metadata_df=metadata_df.iloc[0:2]
currentWaves=listdir(r'C:\Users\Malaz_N\Desktop\waves')

existWavesName=[]
for ele in metadata_df['File name']:
    e=ele[:-4]+'.wav'
    if e in currentWaves:
        existWavesName.append(ele)

temp=pd.DataFrame();
for index, row in metadata_df.iterrows():
    if(row['File name'] in existWavesName):
        temp=temp.append({'File name':row['File name'][:-4] ,'Text':row['Text']},ignore_index=True)
metadata_df=temp        

metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
metadata_df.head(3)


split = int(len(metadata_df) * 0.90)
df_train = metadata_df[:split]
df_val = metadata_df[split:]

print(f"Size of the training set: {len(df_train)}")
print(f"Size of the training set: {len(df_val)}")

# The set of characters accepted in the transcription.
# characters = [' ','ء','إ','أ','آ','ا','ب','ت','ث','ج','ح','خ','د','ذ','ر', 'ز',  'س', 'ش','ص','ض', 'ط', 'ظ','ع', 'غ','ف','ق','ك','ل','م','ن','ه', 'و','ؤ','ي','ى','ئ','ة',' ْ', ' ّ', ' َ', ' ً',' ِ', ' ٌ', ' ُ',' ٍ',]
characters = [ 'َ', 'ط', 'ْ', 'ث', 'م', 'ك', 'ج', 'ض', 'د', 'ي', 'ش', 'ا', 'س', 'ّ', 'خ', ' ', 'غ', 'ُ', 'و', 'ب', 'ل', 'ٍ', 'ذ', 'ى', 'ف', 'ِ', 'ة', 'ق', 'ص', 'ؤ', 'آ', 'ن', 'أ', 'ً', 'ظ', 'ء', 'ه', 'ح', 'ر', 'إ', 'ٌ', 'ع', 'ت', 'ئ', 'ز' ]
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

print(
    f"The vocabulary is: {char_to_num.get_vocabulary()} "
    f"(size ={char_to_num.vocabulary_size()})"
)


# An integer scalar Tensor. The window length in samples.
frame_length = FRAME_LENGTH
# An integer scalar Tensor. The number of samples to step.
frame_step = FRAME_STEP
# An integer scalar Tensor. The size of the FFT to apply.
# If not provided, uses the smallest power of 2 enclosing frame_length.
fft_length = FFT_LENGTH



def encode_single_sample(wav_file, label):
    ###########################################
    ##  Process the Audio
    ##########################################
    # 1. Read wav file
 
    print("wav_file : "+wav_file)
    # print("length is : ",tf.strings.length(wav_file))
    # l=tf.strings.length(wav_file)
    # print("substr : ",tf.strings.substr(wav_file,pos=0,len=l-4))
    # wav_file=tf.strings.substr(wav_file,pos=0,len=l-4);
    
    file = tf.io.read_file(wavs_path + wav_file + ".wav")
    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file,desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    ###########################################
    ##  Process the label
    ##########################################
    # 7. Convert label to Lower case
    label = tf.strings.lower(label)
    # 8. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    # 9. Map the characters in label to numbers
    label = char_to_num(label)
    # 10. Return a dict as our model is expecting two inputs
    return spectrogram, label



batch_size =BATCH_SIZE
# Define the trainig dataset
train_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_train["File name"]), list(df_train["Text"]))
)
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Define the validation dataset
validation_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_val["File name"]), list(df_val["Text"]))
)
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)




fig = plt.figure(figsize=(8, 5))
for batch in train_dataset.take(1):
    spectrogram = batch[0][0].numpy()
    spectrogram = np.array([np.trim_zeros(x) for x in np.transpose(spectrogram)])
    label = batch[1][0]
    # Spectrogram
    label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
    ax = plt.subplot(2, 1, 1)
    ax.imshow(spectrogram, vmax=1)
    ax.set_title(label)
    ax.axis("off")
    # Wav
    file = tf.io.read_file(wavs_path + list(df_train["File name"])[0] + ".wav")
    audio, _ = tf.audio.decode_wav(file)
    audio = audio.numpy()
    ax = plt.subplot(2, 1, 2)
    plt.plot(audio)
    ax.set_title("Signal Wave")
    ax.set_xlim(0, len(audio))
    display.display(display.Audio(np.transpose(audio), rate=16000))
plt.show()



def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss



def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    """Model similar to DeepSpeech2."""
    # Model's input
    input_spectrogram = keras.layers.Input((None, input_dim), name="input")
    # Expand the dimension to use 2D CNN.
    x = keras.layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    # Convolution layer 1
    x = keras.layers.Conv2D(
        filters=FILTERS1,
        kernel_size=KERNEL_SIZE1,
        strides=STRIDES1,
        padding=PADDING1,
        use_bias=False,
        name="conv_1",
    )(x)
    x = keras.layers.BatchNormalization(name="conv_1_bn")(x)
    x = keras.layers.ReLU(name="conv_1_relu")(x)
    # Convolution layer 2
    x = keras.layers.Conv2D(
        filters=FILTERS2,
        kernel_size=KERNEL_SIZE2,
        strides=STRIDES2,
        padding=PADDING2,
        use_bias=False,
        name="conv_2",
    )(x)
    x = keras.layers.BatchNormalization(name="conv_2_bn")(x)
    x = keras.layers.ReLU(name="conv_2_relu")(x)
    # Reshape the resulted volume to feed the RNNs layers
    x = keras.layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    # RNN layers
    for i in range(1, rnn_layers + 1):
        recurrent = keras.layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = keras.layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = keras.layers.Dropout(rate=0.5)(x)
    # Dense layer
    x = keras.layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    x = keras.layers.ReLU(name="dense_1_relu")(x)
    x = keras.layers.Dropout(rate=0.5)(x)
    # Classification layer
    output = keras.layers.Dense(units=output_dim + 1, activation="softmax")(x)
    # Model
    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)
    return model


# Get the model
model = build_model(
    input_dim=fft_length // 2 + 1,
    output_dim=char_to_num.vocabulary_size(),
    rnn_layers=RNN_LAYERS,
    rnn_units=RNN_UNITS,
)

visualkeras.layered_view(model)
model.summary(line_length=110)

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=IS_GREEDY,beam_width=BEAM_WIDTH, top_paths=TOP_PATHS)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


# A callback class to output a few transcriptions during training
class CallbackEval(keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score = wer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)
            
            
            
# Define the number of epochs.
epochs = EPOCHS
# Callback function to check transcription on the val set.
validation_callback = CallbackEval(validation_dataset)
checkpoint_filepath = r'C:\Users\Malaz_N\Desktop\final weights\weights.{epoch:02d}-{val_loss:.2f}.hdf5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='loss',
    mode='min')



# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[validation_callback,model_checkpoint_callback],
)


# Let's check results on more validation samples
predictions = []
targets = []
for batch in validation_dataset:
    X, y = batch
    batch_predictions = model.predict(X)
    batch_predictions = decode_batch_predictions(batch_predictions)
    predictions.extend(batch_predictions)
    for label in y:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        targets.append(label)
wer_score = wer(targets, predictions)
print("-" * 100)
print(f"Word Error Rate: {wer_score:.4f}")
print("-" * 100)
for i in np.random.randint(0, len(predictions), 5):
    print(f"Target    : {targets[i]}")
    print(f"Prediction: {predictions[i]}")

    print("-" * 100)