# Drug-Addicts-Detection-using-Facial-Cues

This project explores AI-driven early detection of drug addiction using facial image analysis.

---
## Dataset Overview

- Kaggle facial image dataset (560 images, balanced: `addicted` and `nonaddicted` classes)
- Datset URL: [https://www.kaggle.com/datasets/ishan8055/drug-addicted-faces](url)
- Data augmentation: rotation, shift, shear, zoom, flip (for robust training)

---

## Architecture

- **Custom CNN:**  
    - Sequential: Multiple Conv2D → BatchNorm → MaxPooling blocks  
    - Dense layers + Dropout, Sigmoid output for binary classification  
- **Transfer Learning:**  
    - VGG16 base (pretrained, frozen) + custom head: GlobalAveragePooling2D → Dense → Dropout → Sigmoid  
- **Deployment:**  
    - Gradio interface for interactive image prediction  
    - Model saved as `.h5` file, reloadable for inference

---

## Workflow

- **Environment Setup**
  
   - Import Deep learning and utility libraries (`TensorFlow/Keras`, `NumPy`, `Matplotlib`, `Seaborn`, `scikit-learn`, `Gradio`).
     
    - Mount Google Drive (Colab):
        ```
        from google.colab import drive
        drive.mount('/content/drive')
        ```

- **Data Loading & Structure**
  
    - Directory:
      Organize dataset in this format
        ```
        addict_dataset/
            drug_addict/
                addicted/
                nonaddicted/
        ```
    - Specify Directories:
        ```
        train_dir = '/content/drive/MyDrive/addict_dataset/drug_addict'
        val_dir   = '/content/drive/MyDrive/addict_dataset/drug_addict'
        test_dir  = '/content/drive/MyDrive/addict_dataset/drug_addict'
        ```
    - Keras `ImageDataGenerator.flow_from_directory` for input pipelines.

- **Preprocessing**
    - Data augmentation and normalization:
        ```
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        ```

- **Model Building**
    - Baseline CNN:
        ```
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
            BatchNormalization(),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2,2),
            Conv2D(128, (3,3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])
        ```

    - VGG16 Transfer Learning:
        ```
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128,128,3))
        for layer in base_model.layers: layer.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        ```

- **Training & Callbacks**
    - Compile and train:
        ```
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(train_data, epochs=25, validation_data=val_data)
        ```
    - Regularization:
        ```
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
        history = model.fit(train_data, epochs=25, validation_data=val_data, callbacks=[early_stop, checkpoint])
        ```

- **Evaluation & Visualization**
    - Performance metrics and plots:
        ```
        test_loss, test_acc = model.evaluate(test_data)
        Y_pred = (model.predict(test_data) > 0.5).astype("int32")
        print(classification_report(test_data.classes, Y_pred))
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.show()
        ```
    - Confusion matrix:
        ```
        cm = confusion_matrix(test_data.classes, Y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()
        ```

    - Sample visualization:
        ```
        x, y = next(test_data)
        preds = (model.predict(x) > 0.5).astype(int)
        plt.figure(figsize=(10,10))
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.imshow(x[i])
            plt.title(f"True: {y[i]}, Pred: {preds[i]}")
            plt.axis('off')
        plt.show()
        ```

- **Model Persistence**
    ```
    model.save('addiction_detector_model.h5')
    model = tf.keras.models.load_model('addiction_detector_model.h5')
    ```

- **Deployment**
    - Gradio web demo:
        ```
        !pip install gradio
        import gradio as gr
        def predict_image(img):
            img = tf.image.resize(img, (128,128))
            img = np.expand_dims(img/255.0, axis=0)
            pred = model.predict(img)
            return "Addicted" if pred > 0.23 else "Non-Addicted"
        gr.Interface(fn=predict_image, inputs="image", outputs="label").launch()
        ```

---

## How to Run

1. Clone repo, upload dataset to Google Drive.
2. Install dependencies, mount drive in Colab.
3. Run notebook step-by-step as per workflow.
4. Launch Gradio for interactive demo.
