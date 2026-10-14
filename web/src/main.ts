import "./ui/theme.css";
import { experiments } from "./app/registry";
import { startApp } from "./app/shell";

startApp(document.getElementById("app")!, experiments);
