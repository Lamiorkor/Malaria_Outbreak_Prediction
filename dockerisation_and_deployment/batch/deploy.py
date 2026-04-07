from train_predict_scheduled import malaria_yearly_train_and_batch_predict

if __name__ == "__main__":
    malaria_yearly_train_and_batch_predict.serve(
        name="malaria-yearly",
        cron="0 9 5 12 *",
    )