![image](https://user-images.githubusercontent.com/91864024/182273839-62299d63-3205-4b63-b938-c5a5d0276a31.png)

# Melbourne House Price Prediction Using ANN 
## I. Outline
- Demand forecasting, house price prediction is always an important requirement. Assuming you are looking for a dream home, you will be very concerned about whether the price offered by the seller is already the lowest? Will the house price be suitable for the market after negotiating? Or you are lucky to have the opportunity to own the best price house in the area because the owner has an urgent need to sell,...
- Home price forecasting will always be helpful to buyers, real estate agents, and sellers also. Because no one wants to buy a house that's 50% or more expensive than the market price. Raising house prices too high only accelerates the freezing of the real estate market, not really helpful in increasing the market's real estate volume.
- In this project, we will build a model that predicts house prices in Melbourne based on house parameters. This can give buyers an idea of ​​a suitable price to negotiate with the seller
## II. Business Objective/ Problem
- Assume that you work in the Data Science department of a real estate company. Your task is to support the buyer and advise the seller of the most suitable price for both parties so that the transaction can be done as soon as possible.
- Your company is expanding to Melbourne, so they urgently need a model to forecast house prices in this area
- This project is built based on that request.
## III. Project implementation
### 1. Business Understanding
Based on the above description => identify the problem:
- Find solutions to attract customers with the most accurate advice, thereby expanding business in this area
- Objectives/problems: build a model to predict house prices based on the parameters of the house, thereby giving suggestions to customers.
- Applied method: ANN
### 2. Data Understanding/ Acquire
- This data was scraped from publicly available results posted every week from Domain.com.au and be cleaned by Tony Pino.
- You can download the dataset at: https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market
- Some Key Details
  - Suburb: Suburb
  - Address: Address
  - Rooms: Number of rooms
  - Price: Price in Australian dollars 
  
  **Method:**
  - S - property sold;
  - SP - property sold prior;
  - PI - property passed in;
  - PN - sold prior not disclosed;
  - SN - sold not disclosed;
  - NB - no bid;
  - VB - vendor bid;
  - W - withdrawn prior to auction;
  - SA - sold after auction;
  - SS - sold after auction price not disclosed.
  - N/A - price or highest bid not available.
 
  **Type:**
  - br - bedroom(s);
  - h - house,cottage,villa, semi,terrace;
  - u - unit, duplex;
  - t - townhouse;
  - dev site - development site;
  - o res - other residential.
  - SellerG: Real Estate Agent
  - Date: Date sold
  - Distance: Distance from CBD in Kilometres
  - Regionname: General Region (West, North West, North, North east …etc)
  - Propertycount: Number of properties that exist in the suburb.
  - Bedroom2 : Scraped # of Bedrooms (from different source)
  - Bathroom: Number of Bathrooms
  - Car: Number of carspots
  - Landsize: Land Size in Metres
  - BuildingArea: Building Size in Metres
  - YearBuilt: Year the house was built
  - CouncilArea: Governing council for the area
  - Lattitude: Self explanitory
  - Longtitude: Self explanitory

![image](https://user-images.githubusercontent.com/91864024/182289145-8099c07d-7261-4fee-bc1d-b1f70d33a3ad.png)

### 3. Build model
#### 3.1. Understand the dataset:

![image](https://user-images.githubusercontent.com/91864024/182282111-8014e61b-b6d1-45bb-b218-07055f5f1a82.png)

![image](https://user-images.githubusercontent.com/91864024/182282172-31405d4a-af2c-4e40-acf6-507b6a01edfc.png)

#### 3.2. Pre-processing data:

![image](https://user-images.githubusercontent.com/91864024/182282918-b90784cd-4ab8-460d-a2aa-2f021bf3707b.png)

![image](https://user-images.githubusercontent.com/91864024/182287339-c2e84a32-4f7d-44d0-b4b1-8f291682ad7d.png)

**Phát hiện và xử lý ngoại lệ**

![image](https://user-images.githubusercontent.com/91864024/182287640-600668c5-63df-48b1-bf0b-6b255895bae8.png)

![image](https://user-images.githubusercontent.com/91864024/182288025-1d118e77-a2e5-4fbe-ac20-f574c21219f9.png)

**Loại bỏ outlier cột Distance**

![image](https://user-images.githubusercontent.com/91864024/182288088-7c725212-fc03-4a41-900d-492aaaf4aa6b.png)

**Chuyển các cột có dạng text sang dạng nhị phân (dummies)**

![image](https://user-images.githubusercontent.com/91864024/182288146-b2e7409d-17c4-4ecb-9b69-204a7d2aeaad.png)

Comment:

Data has preprocessed (df) and Price column still has missing value
-> split df into 2 parts:
- df_final (df has dropped missing value), used to build prediction model
- df_new (df filter to get missing value data), used to predict using the model above

**Loại bỏ NaN df_final**

![image](https://user-images.githubusercontent.com/91864024/182288935-e2e16ef0-8db6-40e6-a31a-7517ace76eff.png)

**Lọc lấy data có NaN để dự đoán**

![image](https://user-images.githubusercontent.com/91864024/182288983-fc7feec4-86df-4e03-8eb9-7d3da764802f.png)

![image](https://user-images.githubusercontent.com/91864024/182280322-7b9356d4-0f44-443d-a993-a8e27c7e504c.png)

#### 3.3 Build model:

**- Split training/testing data**
```python
train_X = df_final.drop(columns=['Price'])
train_y = df_final[['Price']]
```
**- Rescale train_X due to the difference in range between the inputs**
```python
scaler_x = MinMaxScaler()
train_X = scaler_x.fit_transform(train_X)
```
**- Create and add layers to ANN model**
```python
#create model
model = Sequential()
#add model layers
model.add(Dense(182, activation='relu', input_shape=(n_cols, ))) #(362+1)/2
model.add(Dropout(rate=0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.1))
model.add(Dense(128, activation='relu')) #cải tiến bằng cách cho học sâu thêm
model.add(Dense(1, activation='linear')) #output
```
```python
#compile model
model.compile(optimizer='adam', loss='mae')
```
**- Fit model**

```python
from tensorflow.keras.callbacks import EarlyStopping
early_stopping_minitor = EarlyStopping(patience=10)
#train model
history = model.fit(train_X, train_y,
                    epochs=300,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping_minitor])
```
**- Plot loss of train and test set**
```python
print(history.history.keys())
#Loss in train and test
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('model_loss') 
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
```

![image](https://user-images.githubusercontent.com/91864024/182290277-91d8e7c6-1d0a-452d-a4fb-3a4e6a8b5f59.png)

**- Evaluate result**
```python
#evaluate the result
print('Evaluation on test data')
results = model.evaluate(train_X, train_y)
print('mae: ', results)
```

![image](https://user-images.githubusercontent.com/91864024/182290392-93eb8853-7c57-43d8-82d3-5ec595af18d8.png)

Comment: mean(Price) = 9.9e5, mae ~ 180000 --> mae/mean ratio ~ 18% --> acceptable 18% difference from mean to predict house price

#### 3.4. Make prediction on new data
- Use the dataframe df_new that we have filtered with NaN rows. This data uses for new price prediction

**- Remove output column (Price), which contains NaN. The remain columns for inputs**
```python
train_X_1 = df_new.drop(columns=['Price'])
```
**- Transform inputs**
```python
train_X_1 = scaler_x.transform(train_X_1)
```
**- Make prediction on inputs**
```python
test_y_predictions = model.predict(train_X_1)
```
**Print some rows of prediction**
```python
test_y_predictions[:10]
```

![image](https://user-images.githubusercontent.com/91864024/182291443-a286dc46-faba-4137-b738-a39327217c5d.png)

![image](https://user-images.githubusercontent.com/91864024/182292046-f924984d-edaa-42fa-941f-dce120964a64.png)

#### 4. Conclusion

- House price forecasting is a useful tool to help buyers understand the value of the home they are planning to buy
- With a difference of 18% compared to the mean, the model can be used to suggest buying/selling prices to customers

Thank you for your experience with my project. Hope you enjoy it!

