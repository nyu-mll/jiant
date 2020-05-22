class AbstractMetarunner:
    def begin_training(self):
        raise NotImplementedError()

    def yield_train_step(self):
        raise NotImplementedError()

    def should_save_model(self) -> bool:
        raise NotImplementedError()

    def save_model(self):
        raise NotImplementedError()

    def should_save_checkpoint(self) -> bool:
        raise NotImplementedError()

    def save_checkpoint(self):
        raise NotImplementedError()

    def should_eval_model(self) -> bool:
        raise NotImplementedError()

    def eval_model(self):
        raise NotImplementedError()

    def should_break_training(self) -> bool:
        raise NotImplementedError()

    def done_training(self):
        raise NotImplementedError()

    def returned_result(self):
        raise NotImplementedError()

    def run_train_loop(self):
        self.begin_training()

        for _ in self.yield_train_step():
            if self.should_save_model():
                self.save_model()

            if self.should_save_checkpoint():
                self.save_checkpoint()

            if self.should_eval_model():
                self.eval_model()

            if self.should_break_training():
                break

        self.eval_model()
        self.done_training()

        return self.returned_result()
