import Form from "./components/Form";
 import Footer from "./components/Footer";
import Hero from "./components/Hero";
import { useRef } from "react";

function App() {
  const formRef = useRef<HTMLDivElement>(null);
  return (
    <div className="w-full  h-screen justify-between bg-slate-950">
      <Hero formRef={formRef}/>
      <div ref={formRef} className="flex justify-center items-center  ">
        <Form />
      </div>
      <Footer /> 
    </div>
  );
}

export default App;
