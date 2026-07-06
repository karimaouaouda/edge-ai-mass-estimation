# Thesis Updates Requested

## Work Requirements

### Diagrams and Architectural / Explanation Schemas

the system structure and architecture can not be understand only depending on description on by one general schema. the full system architecture must be explained with a set of diagrams and schemas that explain the system structure and architecture in a more clear way.

- `full system architecture diagram`: a diagram that shows the full system architecture, including all components and their interactions. this diagram contains entities each entity represent an interactive component (edge device, host, mqtt server ....) and show the data flow between them. this diagram must be clear and understandable, and must be able to explain the system architecture in a more clear way.

- `some operations diagrams`: this diagrams are sequence diagrams that show the operations of the system, including the interactions between the components and the data flow. and how the messages are exchanged between the components. not all operations, but just the important 4-8 most important operations.


- `use cases diagrams`: this diagrams are use case diagrams that show the use cases of the system, including the actors and the interactions between them. not all use cases, but just the important 4-8 most important use cases.

- `annotation processes diagrams`: this diagram is just activity flow between tools. for example annotation images with segmentation process would be like : 
```
download images -> upload them to annotation tool (cvat)  -> annotate images -> fast review -> save annotated images
```


`notes`:
- the diagrams must be in their places not all diagrams in one section, but each diagram in the section that explain the operation or the component that the diagram is related to.


### Pictures and Screenshots
because the system is a real system, and it is implemented and tested, the thesis must contain pictures and screenshots of the system in action. this will help the reader to understand the system better and to see how it works in practice.

- `tools and technologies screenshots`: in the `thesis_context_folder` i stoere images about tools used, and technologies used in the system. these images must be included in the thesis to show the tools and technologies used in the system.
- `pipelines and operations screenshots`: in the `thesis_context_folder` i store images about the inference and training pipelines, and the operations of the system. these images must be shown in the proper place in the thesis. don't just put all the images in one section, but put them in the section that explain the operation or the component that the image is related to.

- `notes`:
- some important explanations that could have pics i didn't store pics for it, how ever i add a placeholder image in the folder, so if u see that a section need a pic, you can use the placeholder image and add a note that this is a placeholder image and the real image will be added later.

## Layout and Formatting Requirements

after talking to the instructor , the instructor requested to change the order of the sections in the thesis, he requested:
- instead of having  chapter`Experiments and Results` then chapter `Discussion` as a seperated chapters, the teacher to make them under one chapter called `Experiments, Results and Discussion` for example (u can change the name of the chapter to be more suitable, but the idea is to have one chapter that contains both the experiments and results and discussion).

- in the table of content, the appendix A, B looks not good, it looks : 

```tex
A Annotation and Tool Evidence                                           39
A1. Annotation Workflow Screenshot ------------------------------------- 39
A1. Mass Acquisition and Annotation tool ------------------------------- 39
A1. Tool Model Reference ----------------------------------------------- 40

...
```
and same thing for Appendix and it's content in the table of content.

so the instructor require a simple improvement by make them like that:
```latex
Appendix A
Annotation and Tool Evidence
...

Appendix B
....
```

- `note`: this change is only for the table of content, the appendix content will remain the same, just the table of content will be changed to be more clear and readable.