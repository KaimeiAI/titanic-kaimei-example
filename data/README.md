# Data

`train.csv` (891 passengers) and `test.csv` (418 passengers) are the files of Kaggle's getting-started competition [*Titanic - Machine Learning from Disaster*](https://www.kaggle.com/competitions/titanic/data). `test.csv` has no `Survived` column; these notebooks use it as stand-in production traffic.

| Column | Meaning |
| --- | --- |
| `PassengerId` | row id |
| `Survived` | 0 = died, 1 = survived (`train.csv` only) |
| `Pclass` | ticket class: 1 = first, 2 = second, 3 = third |
| `Name` | full name, including a title such as `Mr.`, `Mrs.`, `Miss.` or `Master.` (a boy) |
| `Sex` | `male` or `female` |
| `Age` | age in years; fractional below 1; missing for 177 of the 891 training passengers |
| `SibSp` | siblings and spouses aboard |
| `Parch` | parents and children aboard |
| `Ticket` | ticket number; passengers travelling together often share one |
| `Fare` | price of the ticket in pounds, for **everyone on it** (not per person) |
| `Cabin` | cabin number; mostly missing |
| `Embarked` | port: `C` = Cherbourg, `Q` = Queenstown, `S` = Southampton |

[`src/titanic_kaimei/data.py`](../src/titanic_kaimei/data.py) turns these columns into the v1 and v2 features the notebooks use.

## Source and terms

The underlying passenger list comes from [Encyclopedia Titanica](https://www.encyclopedia-titanica.org/) and is widely redistributed; for example, OpenML lists it as [dataset 40945](https://www.openml.org/d/40945) under a "Public" licence. These two files are Kaggle's split of it, and Kaggle's competition rules cover the files themselves. They're included here so the example runs out of the box; check those terms before reusing them elsewhere.
