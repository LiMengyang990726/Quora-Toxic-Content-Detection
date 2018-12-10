# Quora-Toxic-Content-Detection
This is under Kaggle Competition - Quora toxic content detection

# Highlights from Kernal-FAQ
- Computational Power Limit:

CPU Kernal <= 6hrs

GPU Kernal <= 2hrs

- Submission Format:

Submit through script:

```
submission = pd.DataFrame(
      {'key':test_df.key,'fare_amount':test_y_predictions},
      columns = ['key','fare_amount'])

submission.to_csv('submission.csv', index = False)
```
- Standard Evaluation Method: F1 Score

F1 score = harmonic mean(precision, recall)

precision = true positive / true positive + false positive

recall = true positive / true positive + false negative
- Embeddings: External
