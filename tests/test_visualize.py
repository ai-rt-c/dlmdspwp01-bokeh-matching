from examples.generate_demo_data import build_demo_frames
from mapper import Matcher, assign_test
from visualize import plot_assignment_scatter, plot_fit_panels


def test_bokeh_outputs_are_created(tmp_path):
    train, ideal, test = build_demo_frames()
    matches = Matcher(train, ideal).select_best_ideals()
    assignments = assign_test(ideal, matches, test)
    fit_path = tmp_path / "fit.html"
    assignment_path = tmp_path / "assignment.html"

    plot_fit_panels(train, ideal, matches, str(fit_path))
    plot_assignment_scatter(test, assignments, str(assignment_path))

    assert fit_path.stat().st_size > 1_000
    assert assignment_path.stat().st_size > 1_000
