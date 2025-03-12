import { RefObject } from "react";

interface HeroProps {
  formRef: RefObject<HTMLDivElement>;
}

const Hero: React.FC<HeroProps> = ({ formRef }) => {
  const scrollToForm = () => {
    formRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="py-8 sm:py-16 md:py-24 flex items-center justify-center">
      <div className="mx-auto max-w-[43rem] text-center">
        <p className="text-lg font-medium leading-8 text-gray-500">
          Having trouble with starting up your farms
        </p>
        <h1 className="my-3 sm:my-4 md:my-8 text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight text-white">
        🌾  Predictive tool for <span className="text-blue-700">Farmers </span>
        </h1>
        <p className="text-lg leading-relaxed text-slate-500">
          We help you streamline your workflow by planning, time-boxing, and executing one task at a time.
        </p>

        {/* Scroll to form button */}
        <div className="mt-6 flex items-center justify-center gap-4">
          <button
            onClick={scrollToForm}
            className="inline-block px-6 py-2.5 bg-blue-500 text-white font-medium text-xl leading-tight 
                    hover:bg-blue-600 cursor-pointer hover:shadow-sm focus:bg-blue-600 focus:shadow-sm 
                    focus:outline-none focus:ring-0 active:bg-blue-700 active:shadow-sm disabled:opacity-25 
                    transition duration-150 disabled:pointer-events-none ease-in-out"
          >
            Get started!
          </button>
        </div>
      </div>
    </section>
  );
};

export default Hero;
