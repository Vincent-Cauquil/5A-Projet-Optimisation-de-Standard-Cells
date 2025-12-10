class SequentialPool:
    def __init__(self, tasks):
        self.tasks = tasks

    def run_all(self):
        results = []
        for task in self.tasks:
            result = task.run()
            results.append(result)
        return results