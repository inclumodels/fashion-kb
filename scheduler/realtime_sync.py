"""
APScheduler — runs the scraper automatically every N minutes.
Can be started as a background thread from main.py or the API.
"""

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from ingestion.scraper import run_scraper
from config import SCRAPE_INTERVAL_MINUTES

logger = logging.getLogger(__name__)

_scheduler = None


def start():
    global _scheduler
    if _scheduler and _scheduler.running:
        return

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        _job,
        trigger="interval",
        minutes=SCRAPE_INTERVAL_MINUTES,
        id="fashion_scraper",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info(f"Scheduler started — scraping every {SCRAPE_INTERVAL_MINUTES} min")


def stop():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")


def trigger_now() -> dict:
    """Manually trigger a scrape run and return the summary."""
    return run_scraper()


def _job():
    try:
        summary = run_scraper()
        logger.info(f"Scheduled scrape: {summary}")
    except Exception as e:
        logger.error(f"Scheduled scrape failed: {e}")
