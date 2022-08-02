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

![image](https://user-images.githubusercontent.com/91864024/182280322-7b9356d4-0f44-443d-a993-a8e27c7e504c.png)

### 3. Build model
#### 3.1. Understand the dataset:

![image](https://user-images.githubusercontent.com/91864024/182282111-8014e61b-b6d1-45bb-b218-07055f5f1a82.png)
![image](https://user-images.githubusercontent.com/91864024/182282172-31405d4a-af2c-4e40-acf6-507b6a01edfc.png)

#### 3.2. Pre-processing data:








