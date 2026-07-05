from __future__ import annotations

from datetime import timedelta
import unittest

import app.queue as queue
import app.services.ranking_retrain_scheduler as scheduler
from app.config import RANKER_RETRAIN_DEBOUNCE_SECONDS


class FakeBackgroundTasks:
    def __init__(self):
        self.tasks = []

    def add_task(self, func, *args, **kwargs):
        self.tasks.append({"func": func, "args": args, "kwargs": kwargs})


class FakeArqPool:
    def __init__(self):
        self.calls = []

    async def enqueue_job(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return object()


class RankerRetrainQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_enqueue_ranker_retrain_uses_stable_debounced_job(self):
        pool = FakeArqPool()

        async def fake_get_arq_pool():
            return pool

        original = queue.get_arq_pool
        queue.get_arq_pool = fake_get_arq_pool
        try:
            queued = await queue.enqueue_ranker_retrain("owner-1")
        finally:
            queue.get_arq_pool = original

        self.assertTrue(queued)
        self.assertEqual(len(pool.calls), 1)
        call = pool.calls[0]
        self.assertEqual(call["args"], ("run_ranker_retrain", "owner-1"))
        self.assertEqual(call["kwargs"]["_job_id"], "ranker-retrain:owner-1")
        self.assertEqual(call["kwargs"]["_defer_by"], timedelta(seconds=RANKER_RETRAIN_DEBOUNCE_SECONDS))


class RankerRetrainSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        scheduler.clear_inline_retrain_debounce()

    async def asyncTearDown(self):
        scheduler.clear_inline_retrain_debounce()

    async def test_queued_retrain_does_not_add_inline_background_task(self):
        async def fake_enqueue(owner_id):
            return True

        background_tasks = FakeBackgroundTasks()
        original_enqueue = scheduler.enqueue_ranker_retrain
        scheduler.enqueue_ranker_retrain = fake_enqueue
        try:
            mode = await scheduler.schedule_ranker_retrain("owner-1", background_tasks)
        finally:
            scheduler.enqueue_ranker_retrain = original_enqueue

        self.assertEqual(mode, "queued")
        self.assertEqual(background_tasks.tasks, [])

    async def test_inline_fallback_is_debounced_per_owner(self):
        async def fake_enqueue(owner_id):
            return False

        background_tasks = FakeBackgroundTasks()
        original_enqueue = scheduler.enqueue_ranker_retrain
        scheduler.enqueue_ranker_retrain = fake_enqueue
        try:
            first = await scheduler.schedule_ranker_retrain("owner-1", background_tasks)
            second = await scheduler.schedule_ranker_retrain("owner-1", background_tasks)
            other_owner = await scheduler.schedule_ranker_retrain("owner-2", background_tasks)
        finally:
            scheduler.enqueue_ranker_retrain = original_enqueue

        self.assertEqual(first, "inline")
        self.assertEqual(second, "debounced_inline")
        self.assertEqual(other_owner, "inline")
        self.assertEqual(len(background_tasks.tasks), 2)
        self.assertEqual(background_tasks.tasks[0]["args"], ("owner-1",))
        self.assertEqual(background_tasks.tasks[1]["args"], ("owner-2",))


if __name__ == "__main__":
    unittest.main()
