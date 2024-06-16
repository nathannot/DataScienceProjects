/*
-What devices do my customers use to reach me?
-What product categories am I selling?
-Which product categories do I sell to whom? (Gender Distribution by Category or Product?)
-Which login type do my customers prefer when shopping?
-How does the date and time affect my sales? (Total sales by month, the days of week or time arrival)
-From which product do I earn the most profit per unit?
-How is my delivery speed and order priority?(Delivery Time distribution of order priority by months)
*/
-- 1.What devices do my customers use to reach me?
select 
	Device_Type,
	count(Device_Type)
from retail
group by Device_Type


-- What product categories am I selling?
select
  distinct Product_Category,
  sum(Quantity) as "Total Quantity"
from retail
group by Product_Category

-- Which product categories do I sell to whom?
select
  distinct Product_Category,
   Gender,
  count(Gender) 
from retail
group by Product_Category, Gender

-- Which login type do my customers prefer when shopping?
select
Customer_Login_type,
count(Customer_Login_type)
from retail
group by Customer_Login_type

-- How does the date and time affect my sales? (Total sales by month, the days of week or time arrival)
-- Top 10 Dates
select
	Order_Date,
    sum(Sales) as "Total Sales"
from retail
group by Order_Date
order by sum(Sales) DESC
limit 10

-- top month quanitity sales
select
	month(Order_Date),
    sum(Sales) as "Total Sales"
from retail
group by month(Order_Date)
order by sum(Sales) DESC

-- top week
select
	CONCAT(DATE_FORMAT(MIN(Order_Date), "Y-%m-%d"), " to ", DATE_FORMAT(DATE_ADD(MIN(Order_Date), INTERVAL 6 DAY), "%Y-%m-%d")) AS Week_Dates,
    sum(Sales) as "Total Sales"
from retail
group by week(Order_Date)
order by sum(Sales) DESC
limit 10

ALTER TABLE retail
MODIFY Time TIME;

SELECT
    HOUR(Time) AS Hour_of_Day,
    SUM(Sales) AS Total_Sales
FROM retail
GROUP BY HOUR(Time)
ORDER BY HOUR(Time);

-- -From which product do I earn the most profit per unit?
-- profit per unit
select 
	Product,
    sum(Profit)/sum(Quantity) as "Total Profit per unit"
from retail
group by Product
order by sum(Profit)/sum(Quantity) desc 

-- total profit per product
select 
	Product,
    sum(Profit) as "Total Profit"
from retail
group by Product
order by sum(Profit) desc 

-- -How is my delivery speed and order priority?(Delivery Time distribution of order priority by months)
-- avg delivery time by month and order priority 
select
	Order_Priority,
	month(Order_Date),
   avg(Aging) as "Avg Delivery Time"
from retail
group by  month(Order_Date), Order_Priority