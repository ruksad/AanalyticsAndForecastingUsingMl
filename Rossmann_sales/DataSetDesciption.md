# Dataset Description
    - https://www.kaggle.com/competitions/rossmann-store-sales/overview
You are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the **Sales** column for the test set.

Some stores in the dataset were temporarily closed for refurbishment.

## Files

- **train.csv**: Historical data including `Sales`
- **test.csv**: Historical data excluding `Sales`
- **sample_submission.csv**: Sample submission file in the correct format
- **store.csv**: Supplemental information about the stores

## Data Fields

Most fields are self-explanatory. The table below describes the less obvious ones.

| Field | Description |
| --- | --- |
| Id | An ID representing a `(Store, Date)` tuple within the test set |
| Store | A unique ID for each store |
| Sales | Turnover for any given day (target variable to predict) |
| Customers | Number of customers on a given day |
| Open | Indicator whether a store was open: `0 = closed`, `1 = open` |
| StateHoliday | Indicates a state holiday. Most stores are closed on state holidays. Also note that schools are closed on public holidays and weekends. Values: `a = public holiday`, `b = Easter holiday`, `c = Christmas`, `0 = none` |
| SchoolHoliday | Indicates if the `(Store, Date)` was affected by public school closures |
| StoreType | Store model type: `a`, `b`, `c`, `d` |
| Assortment | Assortment level: `a = basic`, `b = extra`, `c = extended` |
| CompetitionDistance | Distance (in meters) to the nearest competitor store |
| CompetitionOpenSince[Month/Year] | Approximate month/year when the nearest competitor opened |
| Promo | Indicates whether a store is running a promo on that day |
| Promo2 | Ongoing consecutive promotion program: `0 = not participating`, `1 = participating` |
| Promo2Since[Year/Week] | Year and calendar week when the store started participating in Promo2 |
| PromoInterval | Consecutive months when Promo2 starts anew. Example: `Feb,May,Aug,Nov` means each round starts in February, May, August, and November |
