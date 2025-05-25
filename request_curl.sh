curl --location 'localhost:8006/plan' \
--header 'Content-Type: application/json' \
--data '{
  "user_input": "Plan a 7-day trip to a scenic place with hiking and culture under $1800",
  "question": "Is it safe to hike in April?"
}
'