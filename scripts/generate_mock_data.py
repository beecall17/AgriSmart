"""
Generate mock enterprise data for AgriSmart Phase 2.

Creates two files inside the data/ directory:
- data/inventory_db.csv: Realistic agricultural inventory with Nepali warehouse hubs.
- data/logistics_sop.md: Regional transport timelines, weight limits, and hub SOPs.
"""
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INVENTORY_PATH = DATA_DIR / "inventory_db.csv"
LOGISTICS_SOP_PATH = DATA_DIR / "logistics_sop.md"

INVENTORY_ROWS = [
    {
        "product_id": "PRD-001",
        "item_name": "Hybrid Maize Seeds",
        "category": "Seeds",
        "stock_quantity": 2500,
        "warehouse_location": "Kathmandu",
        "unit_price_npr": 120,
    },
    {
        "product_id": "PRD-002",
        "item_name": "Rice Seeds (IRRI-6)",
        "category": "Seeds",
        "stock_quantity": 1800,
        "warehouse_location": "Pokhara",
        "unit_price_npr": 85,
    },
    {
        "product_id": "PRD-003",
        "item_name": "Wheat Seeds (Lokwan)",
        "category": "Seeds",
        "stock_quantity": 3200,
        "warehouse_location": "Biratnagar",
        "unit_price_npr": 95,
    },
    {
        "product_id": "PRD-004",
        "item_name": "Urea Fertilizer",
        "category": "Fertilizer",
        "stock_quantity": 5000,
        "warehouse_location": "Kathmandu",
        "unit_price_npr": 45,
    },
    {
        "product_id": "PRD-005",
        "item_name": "DAP Fertilizer",
        "category": "Fertilizer",
        "stock_quantity": 4200,
        "warehouse_location": "Biratnagar",
        "unit_price_npr": 60,
    },
    {
        "product_id": "PRD-006",
        "item_name": "Potash",
        "category": "Fertilizer",
        "stock_quantity": 1500,
        "warehouse_location": "Pokhara",
        "unit_price_npr": 75,
    },
    {
        "product_id": "PRD-007",
        "item_name": "Organic Compost",
        "category": "Fertilizer",
        "stock_quantity": 900,
        "warehouse_location": "Kathmandu",
        "unit_price_npr": 120,
    },
    {
        "product_id": "PRD-008",
        "item_name": "Organic Pesticide",
        "category": "Pesticide",
        "stock_quantity": 600,
        "warehouse_location": "Kathmandu",
        "unit_price_npr": 450,
    },
    {
        "product_id": "PRD-009",
        "item_name": "Fungicide (Carbendazim)",
        "category": "Pesticide",
        "stock_quantity": 350,
        "warehouse_location": "Pokhara",
        "unit_price_npr": 320,
    },
    {
        "product_id": "PRD-010",
        "item_name": "Drip Irrigation Kit",
        "category": "Irrigation",
        "stock_quantity": 120,
        "warehouse_location": "Kathmandu",
        "unit_price_npr": 25000,
    },
    {
        "product_id": "PRD-011",
        "item_name": "Sprinkler Set",
        "category": "Irrigation",
        "stock_quantity": 280,
        "warehouse_location": "Biratnagar",
        "unit_price_npr": 8500,
    },
    {
        "product_id": "PRD-012",
        "item_name": "Cattle Feed",
        "category": "Animal Husbandry",
        "stock_quantity": 4000,
        "warehouse_location": "Biratnagar",
        "unit_price_npr": 150,
    },
]

INVENTORY_COLUMNS = [
    "product_id",
    "item_name",
    "category",
    "stock_quantity",
    "warehouse_location",
    "unit_price_npr",
]

LOGISTICS_SOP_MD = """# AgriSmart Logistics Standard Operating Procedures

## 1. Regional Transport Timelines

The following are standard road-transit estimates from the central Kathmandu
distribution hub to regional depots under normal weather and road conditions.

| Route | Distance (approx.) | Standard Transit Time | Priority Transit Time |
|-------|--------------------:|----------------------:|----------------------:|
| Kathmandu → Pokhara | ~200 km | 6–8 hours | 5 hours |
| Kathmandu → Biratnagar | ~400 km | 8–10 hours | 7 hours |
| Kathmandu → Chitwan | ~150 km | 4–5 hours | 3.5 hours |
| Pokhara → Biratnagar | ~600 km | 12–14 hours | 10 hours |
| Biratnagar → Chitwan | ~500 km | 10–12 hours | 8 hours |

> **Note:** Transit times increase by 20–40% during monsoon season (June–September)
> or during political hartal (strike) days.

---

## 2. Vehicle Weight Limits

All dispatch requests must comply with the following gross vehicle weight (GVW)
limits. Exceeding these limits requires a special overweight permit and may void
insurance coverage.

| Vehicle Type | Max Load (kg) | Typical Use Case |
|-------------|--------------:|-----------------|
| Mini Truck (3-wheeler / small 4-wheeler) | 3,000 kg | Urban deliveries, small seed/pesticide orders |
| Medium Truck (10-wheeler) | 10,000 kg | Standard fertilizer and irrigation kit orders |
| Heavy Truck (12–14-wheeler) | 20,000 kg | Bulk seed, large irrigation systems, warehouse transfers |

### Product-specific loading rules

- **Bulk commodities** (Urea, DAP, Potash, Cattle Feed): Must be palletized;
  max 10,000 kg per medium truck.
- **Hazardous materials** (Pesticides, Fungicides): Must be segregated from
  food-grade seeds and feeds; max 3,000 kg per trip unless IMDG-compliant
  packaging is used.
- **Irrigation kits** (Drip, Sprinkler): Freight is calculated by volumetric
  weight; bulky kits count as 5x physical weight for truck-assignment purposes.

---

## 3. Hub Standard Operating Procedures

### 3.1 Receiving & Inbound Inspection

1. All inbound shipments must be accompanied by a Delivery Note (DN) and,
   where applicable, a Material Safety Data Sheet (MSDS).
2. Warehouse staff verify: product ID, batch/lot number, expiry date (for
   pesticides and seeds), and physical condition of packaging.
3. Discrepancies (shortage, damage, wrong SKU) must be logged in the
   Warehouse Discrepancy Register within 24 hours.
4. Pesticides and fertilizers must be quarantined in a separate, ventilated
   bay pending storage allocation.

### 3.2 Storage & Climate Control

- **Seeds:** Store at 15–20 °C, relative humidity 40–60%. Rotate stock using
  FIFO (First In, First Out).
- **Fertilizers:** Keep dry; protect from direct rain. Urea is hygroscopic
  and must be stacked on pallets with plastic sheeting.
- **Pesticides:** Store in a locked, temperature-controlled room (5–30 °C).
  Maintain a separate spill-containment tray under every shelf.
- **Irrigation equipment:** Store in a covered, dust-free area. Kits with
  electronic components must be kept above floor level.

### 3.3 Dispatch & Order Fulfillment

1. Field-agent requests entered via AgriSmart are forwarded to the nearest
   hub within 30 minutes during business hours (09:00–17:00 NPT).
2. If stock is insufficient, the hub manager must notify the requesting agent
   within 2 hours and propose an alternative hub or substitute product.
3. Orders with a combined value > NPR 500,000 require secondary approval from
   the regional logistics coordinator.
4. All dispatches require: (a) signed Delivery Note, (b) vehicle registration
   copy, and (c) driver identification.

### 3.4 Safety & Compliance

- Drivers transporting pesticides must wear PPE (gloves, mask) and carry a
  spill kit.
- Emergency contact numbers for each hub must be displayed prominently inside
  the warehouse office.
- All incidents (spills, accidents, theft) must be reported to the regional
  coordinator within 1 hour.

---

## 4. Contact Directory

| Hub | Address | Emergency Contact |
|-----|---------|------------------|
| Kathmandu Central | Thimi, Bhaktapur | +977-1-6631234 |
| Pokhara Regional | Lakeside, Pokhara-6 | +977-61-465123 |
| Biratnagar Eastern | Roadways, Biratnagar-12 | +977-21-587234 |
"""

def generate_inventory_csv() -> None:
    """Write the inventory CSV to the data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with INVENTORY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(INVENTORY_ROWS)

    print(f"[OK] Inventory CSV written to: {INVENTORY_PATH}")


def generate_logistics_sop() -> None:
    """Write the logistics SOP markdown to the data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    LOGISTICS_SOP_PATH.write_text(LOGISTICS_SOP_MD, encoding="utf-8")

    print(f"[OK] Logistics SOP written to: {LOGISTICS_SOP_PATH}")


def main() -> None:
    """Generate all mock enterprise data files."""
    print("Generating mock data for AgriSmart Phase 2...")

    generate_inventory_csv()
    generate_logistics_sop()

    print("\n[Summary]")
    print(f"  Inventory rows    : {len(INVENTORY_ROWS)}")
    print(f"  Inventory columns : {len(INVENTORY_COLUMNS)}")
    print("  Done.")


if __name__ == "__main__":
    main()