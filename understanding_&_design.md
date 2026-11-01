# Accountant.ai — understanding and design

Notes for the first build. The product is a personal accountant for anyone, including someone with $500 and someone with no stable place to live. The hard part is the model. Bank connections, hosting, and live transfers come after the model can judge a file.

The model is called Accountant.ai. Version 0.0 is a decision model trained in Colab. A DigitalOcean droplet is only for inference, later, and it does not train.

## What the accountant is

It opens one person’s file and writes an opinion before money moves. The file can be a single number or a full set of accounts. There is no minimum balance and no bank is required to start.

Two questions sit under every opinion:

1. Can this be paid from money already marked spendable?
2. Should it be paid once everything else they have, owe, and need is counted?

The person still decides. The model cannot stop a human choice. It can show what the choice keeps, what it maintains, and what it changes later.

It is not a habit tracker and it is not a daily-routines app. It does not try to delete spending. Things cost money. The useful questions are:

- What was kept today?
- Was money used to maintain something?
- What can be spent, and what cannot?

## Spendable, kept, and maintained

| Kind | Meaning | How far it reaches |
| --- | --- | --- |
| Able to spend | Inside the amount already accepted for this period | Changes this week. Becomes a long effect only if it keeps happening |
| Not able to spend | Protected, already promised, or needed today | Shows up fast: a bill, the floor, or today’s need |
| Saved | Money that stayed because it was not used | “Did they keep any today?” |
| Maintain | Money that kept something current: phone, shelter, a debt, an asset, groceries from a food benefit | Skipping it is not a save |

Repeating charges are described as what they are: bus fare, food, the phone bill. A repeating amount that stays protected is a save. One purchase is not turned into a personality. A day with no transactions stays unknown. “Complete” means every dollar in the file is classed and the gaps are named.

### Down the road

The same choice is mapped forward. The map is an inference, not a fact.

- **Days.** What is left to spend, what was kept, and whether a bill is closer or covered.
- **Months.** Only if the same choice repeats. A one-time spend stays a one-time spend.
- **Years.** Only from a pattern already in the file, or from a choice that touches protected money.

A training row is wrong when it treats every outflow as a habit to cut, calls a maintain payment a failure to save, or stretches one coffee into a year.

## Money movement, without issuing a card

We do not issue a debit card. The person uses accounts they already have. From the start, every outflow the accountant is responsible for is a request.

- **Block** is the default. Silence is a block.
- **Accept** is an explicit yes after the file has been checked. That yes is what allows a move.
- **Move** is a transfer between two accounts they own, or a bill they named. Their bank moves the money. We never store the bank password. Looking and moving are separate permissions.

The card they shop with stays on a spending account that only holds money already accepted and moved. The rest stays in a protected account. That is the block that works without issuing a card. A swipe on the protected account can still go through if they use that card. Cash in a pocket cannot be frozen. Someone with one account, or with cash only, still gets the request list immediately. A real transfer waits until there is a second account or a bill to send it to.

We see balances and transactions through a read-only bank connection or a statement they upload. They can also type what they have. An unconnected account is named as missing.

## Government help, by place

When income stops, or food and shelter do not fit in the file, the map adds local assistance beside the money. The model does not decide that the person qualifies and it does not invent a monthly amount. It names the programs for that place, the official door, and the facts still missing. The agency decides the award.

### Facts required first

| Fact | Why it changes the answer |
| --- | --- |
| State where they live | SNAP, cash aid, Medicaid, and energy help are applied for in the state of residence |
| State where they worked | Unemployment is filed where the job was, which can differ from home |
| County or city | Food pantries, shelter, and utility help are local |
| Household size | Food-benefit tables are per household |
| What stopped | A lost job points at unemployment. Low or no income points at food, health, and cash aid |

If any of those are missing, the model asks. It does not fill them in.

### Job loss

The file records that pay stopped and the date. The days ahead show what cash remains and which bills are maintain, not optional. Unemployment and food benefits are not written in as money already received.

Beside that, the model lists where to apply:

- Unemployment, in the state where they worked, through the [CareerOneStop unemployment finder](https://www.careeronestop.org/localhelp/unemploymentbenefits/Find-unemployment-benefits.aspx).
- SNAP, in the state where they live, through that state’s own form in the [SNAP state directory](https://www.fna.usda.gov/snap/state-directory). Eligibility rules are published at [SNAP eligibility](https://www.fna.usda.gov/snap/recipient/eligibility). The federal office does not process applications or see case files.
- Cash assistance, Medicaid, and other state help through [USAGov state social services](https://www.usa.gov/state-social-services).
- A pantry, shelter, or utility help nearby through [211](https://www.211.org/).
- The broader index is [USAGov benefits](https://www.usa.gov/benefits).

A food benefit, once the agency actually issues it, is maintain money for groceries. It is not a spending habit, and it is not cash the app can move onto a debit card.

Dollar figures live in a dated table taken from those official pages, with the inputs the table needs, such as household size. The model may quote that table. It may not remember a number from training and present it as what this person will get.

A training row is wrong when it says “you will get” a benefit, or when it tells the person to leave a fact off an application.

## v0.0 — the decision dataset

v0.0 trains one behavior: given a file, a rule, and a request, answer **block**, **accept**, **change**, or **ask**. The reason includes the class of the money (spent, saved, or maintain), what was kept, the three time horizons, and, when the situation calls for it, the local help list with gaps named.

### Row shape

The same keys are shared with a separate inference project that uses different information. This run trains only `task: accountant`. Mixing the other information into this file teaches the wrong job. The same notebook can train the other model later by pointing it at the other file.

```json
{
  "task": "accountant",
  "file": {},
  "rule": "help me save this",
  "request": {},
  "decision": "block",
  "reason": "",
  "money_class": "save",
  "kept": "",
  "horizon": { "days": "", "months": "", "years": "" },
  "local_help": []
}
```

`file` holds the picture: cash, what is protected, what is for today, bills, and any place facts already known. `local_help` is empty unless income stopped or a basic need does not fit. Each help item is an official door, a date, and the facts still missing. It is not an award.

### Data to train on

| Share | Set | Source |
| --- | --- | --- |
| About 80% | Decision rows. The program computes the label. A second check confirms it. Disagreements are dropped. A model does not write the labels. | Generated |
| About 10% | Ask or refuse when the amount, the account, or the place is missing | Same generator. The useful habit from [CashSage](https://www.kaggle.com/datasets/jainamshahhh/cashsage-personal-finance-computation) is to ask instead of inventing a number. CashSage itself is payoff math, not this ledger |
| About 10% | Plain speech, so the reason sounds like a person | A thin slice of [Kuvera PersonalFinance V2.1](https://huggingface.co/datasets/Akhil-Theerthala/Kuvera-PersonalFinance-V2.1) (Apache-2.0). Those answers are advice, not our labels |

Leave out FinGPT, PIXIU, and other market models. Leave out real bank logins and real statements.

The generator must cover at least:

- An impulse spend against a save rule is blocked, or changed to the weekly amount.
- Food, a bus, a phone, an ID, or shelter, paid from money set aside for today, is accepted.
- A want that would spend today’s survival money is blocked.
- A missing amount, account, or place is a question.
- A move inside the weekly allowance and the floor is accepted.
- A move larger than the protected cash is blocked.
- Silence is a block.
- Wealth on paper with not enough cash is a change. The model does not invent a sale.
- A lost job adds the resource list and does not book benefits as cash on hand.
- A maintain payment is not labeled a failure to save.

### Research to read before Colab

- CashSage’s write-up. Labels come from a program, then a second program checks them.
- Kuvera’s paper, *Synthesizing Behaviorally-Grounded Reasoning Chains*. Useful for how a short reason is written. Their answers are not our labels.
- Unsloth’s [Qwen3 fine-tune guide](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune).

### Required before a training run

- The row keys stay frozen.
- The generator passes the cases above.
- 300 rows are held out and never trained on.
- No account numbers, passwords, or real statements in the files.
- Base model: Qwen3-8B, Apache-2.0, 4-bit, adapter only. `unsloth/Qwen3-8B` in 4-bit, sequence length 2048. If a free Colab T4 runs out of memory, shorten that to 1024 or train Qwen3-4B on the same rows.
- The held-out score matches the decision, uses the file’s numbers in the reason, asks when a fact is missing, and never says the money or a benefit already moved.

v0.0 is finished when that score holds. The bank connector, the droplet, and live transfers wait until then.

## Hosting

Colab trains the adapter. When the score passes, a DigitalOcean droplet loads Qwen3-8B plus that adapter and answers one row at a time. Training stays in the notebook.

## What already exists

Nothing already does this whole job. Checked 1 November 2026.

- **Books.** [Actual Budget](https://github.com/actualbudget/actual) is the mature MIT app that starts from a typed balance and uses envelopes. It does not block a card or pay an allowance. Firefly III, Sure, and hledger can keep books, and using their code would put this app under a stricter license.
- **Ask first.** [Ledger](https://github.com/zenatron/ledger) is a small AGPL app whose product is “ask before you buy.” Copy that rule. Write our own code.
- **Card.** We are not issuing one. FamZoo’s pattern (most money off the spending card until someone says yes) is the behavior to copy on accounts the person already has. A card network will not wait for a conversation: Stripe needs an answer in about 2 seconds. Decline-or-wait happens in our request list, before the move.

## Order of work

1. Freeze the row and write the labeler tests.
2. Generate the dataset, including the local-help cases.
3. Fine-tune the adapter in Colab.
4. Score the held-out 300.
5. Only then stand up the droplet for inference.
