from ortools.sat.python import cp_model
import pandas as pd
import joblib
from datetime import timedelta
import math

EF_CO2 = 3.114

def predict_turnaround_in_hours():
    df = pd.read_csv(
        "notebooks/data/test_input.csv",
        dtype={
            "imo": "int32",
            "company_name": "object",
            "ship_type": "object",
            "gt": "int64",
            "dwt": "int64",
            "length": "int64",
            "width": "int64",
            "age": "int64",
            "fuel_consumption": "float64",
        },
        parse_dates=["eta"]
    )

    model = joblib.load("notebooks/models/trained.pkl")

    features = [
        "gt", "dwt", "length", "width", "age",
        "fuel_consumption",
        "ship_type", "company_name",
        "median_time_in_port_hours",
        "tech_eff_index",
        "tech_eff_value",
    ]

    df["pred_turnaround_h"] = model.predict(df[features])

    return df

def calculate_co2_rate_tph(df):
    df["fuel_tph_alongside"] = df["fuel_consumption"] / 24 * 0.15
    df["co2_rate_tph"] = df["fuel_tph_alongside"] * EF_CO2
    return df


def plan(number_of_berths):
    df = predict_turnaround_in_hours()
    df = calculate_co2_rate_tph(df)

    # Variable definition
    model = cp_model.CpModel()
    time_unit_min = 10
    gap_slots = 1  # one 10-min slot buffer
    horizon_start = df["eta"].min().floor("h")
    horizon_end = horizon_start + timedelta(
        hours=df["pred_turnaround_h"].sum() / number_of_berths + 48
    )

    total_minutes = (horizon_end - horizon_start).total_seconds() / 60
    horizon_slots = math.ceil(total_minutes / time_unit_min)

    pres = {}  # pres[i,b] = BoolVar: “ship i uses berth b?”
    start = {}  # start[i,b] = IntVar: berth‐start slot if pres==1
    interval = {}  # interval[i,b] = OptionalIntervalVar(start, dur, start + dur, pres)

    for i, ship in df.iterrows():
        dur_slots = int(ship["pred_turnaround_h"] * 60 / time_unit_min)
        eta_off = math.ceil(
            (ship["eta"] - horizon_start).total_seconds()
            / (60 * time_unit_min)
        )

        pres[i] = {}
        start[i] = {}
        interval[i] = {}
        for b in range(number_of_berths):
            pres[i][b] = model.NewBoolVar(f"pres_s{i}_b{b}")

            start[i][b] = model.NewIntVar(
                eta_off, horizon_slots - dur_slots - gap_slots,
                f"start_s{i}_b{b}"
            )
            # extend interval by gap_slots for no-overlap
            interval[i][b] = model.NewOptionalIntervalVar(
                start[i][b],
                dur_slots + gap_slots,
                start[i][b] + dur_slots + gap_slots,
                pres[i][b],
                f"int_s{i}_b{b}"
            )


    # Constraints definition
    for i in pres:
        model.Add(sum(pres[i].values()) == 1)

    for b in range(number_of_berths):
        model.AddNoOverlap([interval[i][b]
                            for i in pres
                            if b in pres[i]])


    # Build the objective function
    wait_penalty_hr = 1000  # $/hour
    co2_price_per_t = 80  # $ per tonne CO₂

    wait_penalty_slot = int(wait_penalty_hr * time_unit_min / 60)

    wait_slots = {}
    for i, ship in df.iterrows():
        wait_slots[i] = {}
        eta_off = math.ceil(
            (ship["eta"] - horizon_start).total_seconds()
            / (60 * time_unit_min)
        )
        for b in pres[i]:
            w = model.NewIntVar(0, horizon_slots, f"wait_s{i}_b{b}")
            wait_slots[i][b] = w
            model.Add(w == start[i][b] - eta_off).OnlyEnforceIf(pres[i][b])
            model.Add(w == 0).OnlyEnforceIf(pres[i][b].Not())

    cii_score = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
    cii_weight = 100

    obj_terms = []
    for i, ship in df.iterrows():
        co2_cost = int(ship["pred_turnaround_h"]
                       * ship.co2_rate_tph
                       * co2_price_per_t)

        score = cii_score.get(ship["cii_rating"], 0)

        for b in pres[i]:
            p = pres[i][b]
            # waiting cost: slots * $/slot
            obj_terms.append(wait_slots[i][b] * wait_penalty_slot)
            # eco cost: constant * bool
            obj_terms.append(co2_cost * p)
            # c) CII “bonus” → subtract to reward green ships
            #    (A=4 → −400, B=3 → −300, … E=0 → 0)
            obj_terms.append(- score * cii_weight * p)

    model.Minimize(sum(obj_terms))

    # Solve the model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule found")

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("Solution found.")
        print("Schedule:")
        for i, ship in df.iterrows():
            if i in pres:
                for b in range(number_of_berths):
                    if solver.Value(pres[i][b]):
                        start_time = horizon_start + timedelta(minutes=solver.Value(start[i][b]) * time_unit_min)
                        dur_slots = int(ship["pred_turnaround_h"] * 60 / time_unit_min)
                        end_time = start_time + timedelta(minutes=dur_slots * time_unit_min)
                        wait_time = solver.Value(wait_slots[i][b]) * time_unit_min / 60
                        print(
                            f"Ship {i} assigned to Berth {b}: Start={start_time.strftime('%Y-%m-%d %H:%M')}, End={end_time.strftime('%Y-%m-%d %H:%M')}, Wait={wait_time:.2f} hours")

        print(f"Objective value: {solver.ObjectiveValue()}")


if __name__ == "__main__":
    number_of_berths = 3
    plan(number_of_berths)




