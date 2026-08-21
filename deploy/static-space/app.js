const prompts = {
  accurate: {
    prompt: "Describe only what is visibly supported by this astronomy image. Do not invent mission names, locations, measurements, or dates.",
    output: "A rust-toned planetary surface stretches toward a hazy horizon, with layered terrain and scattered rocky formations visible in the foreground."
  },
  objects: {
    prompt: "List the main astronomical objects or visible structures in this image, then provide one concise caption.",
    output: "Visible structures include a broad rocky foreground, a gently layered horizon, and a small illuminated body against the dark sky."
  },
  cautious: {
    prompt: "Describe this image conservatively. Separate direct visual evidence from anything uncertain.",
    output: "The image appears to show a reddish rocky landscape beneath a dark sky. The exact location and identity of the distant bright object cannot be confirmed from visual evidence alone."
  }
};

const imageInput = document.querySelector("#imageInput");
const imagePreview = document.querySelector("#imagePreview");
const imageLabel = document.querySelector("#imageLabel");
const dropzone = document.querySelector("#dropzone");
const promptInput = document.querySelector("#promptInput");
const outputBox = document.querySelector("#outputBox");
const outputText = document.querySelector("#outputText");
const generateButton = document.querySelector("#generateButton");
const resetButton = document.querySelector("#resetButton");
const presetButtons = [...document.querySelectorAll(".preset")];

let activePreset = "accurate";

function choosePreset(name) {
  activePreset = name;
  promptInput.value = prompts[name].prompt;
  presetButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.prompt === name);
  });
}

function previewFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const reader = new FileReader();
  reader.addEventListener("load", () => {
    imagePreview.src = reader.result;
    imageLabel.textContent = `LOCAL PREVIEW · ${file.name.toUpperCase()}`;
    dropzone.classList.add("has-image");
  });
  reader.readAsDataURL(file);
}

imageInput.addEventListener("change", () => previewFile(imageInput.files[0]));
presetButtons.forEach((button) => {
  button.addEventListener("click", () => choosePreset(button.dataset.prompt));
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
});
dropzone.addEventListener("drop", (event) => previewFile(event.dataTransfer.files[0]));

generateButton.addEventListener("click", () => {
  outputBox.classList.add("loading");
  generateButton.disabled = true;
  generateButton.querySelector("span").textContent = "Preparing UI preview…";

  window.setTimeout(() => {
    outputText.textContent = prompts[activePreset].output;
    outputBox.classList.remove("loading");
    generateButton.disabled = false;
    generateButton.querySelector("span").textContent = "Generate preview";
  }, 900);
});

resetButton.addEventListener("click", () => {
  imageInput.value = "";
  imagePreview.removeAttribute("src");
  dropzone.classList.remove("has-image");
  imageLabel.textContent = "SAMPLE · MARS SURFACE";
  choosePreset("accurate");
  outputText.textContent = prompts.accurate.output;
  outputBox.classList.remove("loading");
});
