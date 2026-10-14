import asyncio

from textual.widgets import Static

from geneweaver.dashboard import GeneWeaverApp


def test_dashboard_compares_the_sample_fasta_files() -> None:
    async def exercise_dashboard() -> tuple[str, str]:
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            await pilot.click("#compare")
            for _ in range(50):
                await pilot.pause(0.1)
                status = str(app.query_one("#status", Static).render())
                if status == "Comparison complete." or status.startswith("Error:"):
                    break
            results = str(app.query_one("#results", Static).render())
            return status, results

    status, results = asyncio.run(exercise_dashboard())

    assert status == "Comparison complete."
    assert "Sequences: 4 | chunks processed: 4" in results
    assert "exact_match: length=16, mismatches=0, positions: none" in results
    assert "two_substitutions: length=16, mismatches=2, positions: 4, 15" in results
