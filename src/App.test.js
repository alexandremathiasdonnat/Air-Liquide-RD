import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";

test("renders the app shell and aggregation controls", () => {
  render(<App />);
  expect(screen.getByText(/MoE Runner/i)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Opera BOA/i })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /HMOE BOA/i })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Comparaison par simulation de Monte Carlo/i })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Monte Carlo Gridsearch par méthode/i })).toBeInTheDocument();
});
test("shows Monte Carlo per-algorithm parameter sources", async () => {
  render(<App />);

  await userEvent.click(screen.getByRole("button", { name: /Opera FTRL/i }));
  await userEvent.click(screen.getByRole("button", { name: /Run/i }));
  expect(await screen.findByRole("button", { name: /Export CSV/i })).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: /Comparaison par simulation de Monte Carlo/i }));

  expect(screen.getAllByText(/Dernier run propre/i).length).toBeGreaterThan(0);
  expect(screen.getAllByText(/eta0 0\.01/i).length).toBeGreaterThan(0);
  expect(screen.getAllByText(/Valeurs par defaut/i).length).toBeGreaterThan(0);
});

test("runs an HMOE aggregation from the UI", async () => {
  render(<App />);

  await userEvent.click(screen.getByRole("button", { name: /HMOE BOA/i }));
  await userEvent.click(screen.getByRole("button", { name: /Run/i }));

  expect(await screen.findByText(/Regimes HMOE actifs/i)).toBeInTheDocument();
});

test("shows Monte Carlo prerequisites", async () => {
  render(<App />);

  await userEvent.click(screen.getByRole("button", { name: /Comparaison par simulation de Monte Carlo/i }));

  expect(screen.getByText(/mode Aléatoire est actif/i)).toBeInTheDocument();
  expect(screen.getByText(/figer les conditions de génération de référence/i)).toBeInTheDocument();
});
