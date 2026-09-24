import http from "k6/http";
import { check } from "k6";
import { Counter, Rate } from "k6/metrics";

const successfulRequests = new Counter("birdly_successful_requests");
const errorRequests = new Counter("birdly_error_requests");
const errorRate = new Rate("birdly_error_rate");

const imageBytes = open(__ENV.IMAGE_PATH, "b");
const imagePart = http.file(
  imageBytes,
  __ENV.IMAGE_NAME,
  __ENV.IMAGE_CONTENT_TYPE,
);

const virtualUsers = Number(__ENV.VUS);
export const options = __ENV.ITERATIONS
  ? {
      vus: virtualUsers,
      iterations: Number(__ENV.ITERATIONS),
    }
  : {
      vus: virtualUsers,
      duration: __ENV.DURATION,
    };

export default function () {
  const response = http.post(
    `${__ENV.API_BASE_URL}/predict`,
    { image: imagePart },
    {
      tags: { name: "POST /predict" },
      timeout: __ENV.REQUEST_TIMEOUT,
    },
  );

  let validPrediction = false;
  if (response.status === 200) {
    try {
      const predictions = response.json();
      validPrediction = Array.isArray(predictions) && predictions.length === 5;
    } catch (_) {
      validPrediction = false;
    }
  }
  const successful = check(response, {
    "status is 200": (result) => result.status === 200,
    "returns five predictions": () => validPrediction,
  });
  successfulRequests.add(successful ? 1 : 0);
  errorRequests.add(successful ? 0 : 1);
  errorRate.add(!successful);
}
