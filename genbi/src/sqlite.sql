--
-- SQLite database initialisation script.
--
-- @author Lawrence Murray <lawrence.murray@csiro.au>
-- $Rev$
-- $Date$
--

CREATE TABLE IF NOT EXISTS Trait (
  Name VARCHAR PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS Category (
  Name VARCHAR PRIMARY KEY,
  Description VARCHAR,
  Position INTEGER UNIQUE
);

CREATE TABLE IF NOT EXISTS Node (
  Name VARCHAR PRIMARY KEY,
  Description VARCHAR,
  Category VARCHAR NOT NULL REFERENCES Category(Name),
  HasX BOOLEAN DEFAULT FALSE,
  HasY BOOLEAN DEFAULT FALSE,
  HasZ BOOLEAN DEFAULT FALSE,
  Position INTEGER
);

CREATE TABLE IF NOT EXISTS NodeTrait (
  Node VARCHAR NOT NULL REFERENCES Node(Name),
  Trait VARCHAR NOT NULL REFERENCES Trait(Name),
  PRIMARY KEY (Node, Trait)
);

CREATE TABLE IF NOT EXISTS NodeFormula (
  Node VARCHAR NOT NULL REFERENCES Node(Name),
  Function VARCHAR NOT NULL,
  Formula VARCHAR NOT NULL,
  XOrdinate INTEGER DEFAULT -1,
  YOrdinate INTEGER DEFAULT -1,
  ZOrdinate INTEGER DEFAULT -1,
  Position INTEGER,
  PRIMARY KEY(Node, Function, XOrdinate, YOrdinate, ZOrdinate)
);

CREATE TABLE IF NOT EXISTS Edge (
  ParentNode VARCHAR NOT NULL REFERENCES Node(Name),
  ChildNode VARCHAR NOT NULL REFERENCES Node(Name),
  XOffset INTEGER DEFAULT 0,
  YOffset INTEGER DEFAULT 0,
  ZOffset INTEGER DEFAULT 0,
  Position INTEGER,
  PRIMARY KEY (ParentNode, ChildNode, XOffset, YOffset, ZOffset)
);

-- Flattened list of parents, inlining intermediate results
CREATE TABLE IF NOT EXISTS Parent (
  ParentNode VARCHAR NOT NULL REFERENCES Node(Name),
  ChildNode VARCHAR NOT NULL REFERENCES Node(Name),
  XOffset INTEGER DEFAULT 0,
  YOffset INTEGER DEFAULT 0,
  ZOffset INTEGER DEFAULT 0,
  Position INTEGER,
  UNIQUE (ParentNode, ChildNode, XOffset, YOffset, ZOffset)
);

--
-- Foreign keys
--

-- Node.Category -> Category.Name
CREATE TRIGGER IF NOT EXISTS NodeInsert AFTER INSERT ON Node
  WHEN
    (SELECT 1 FROM Category WHERE Name = NEW.Category) IS NULL
  BEGIN
    SELECT RAISE(ABORT, 'Category does not exist');
  END;
    
-- NodeTrait.Node -> Node.Name
CREATE TRIGGER IF NOT EXISTS NodeTraitNodeInsert AFTER INSERT ON NodeTrait
  WHEN
    (SELECT 1 FROM Node WHERE Name = NEW.Node) IS NULL
  BEGIN
    SELECT RAISE(ABORT, 'Node does not exist');
  END;

-- NodeTrait.Trait -> Trait.Name
CREATE TRIGGER IF NOT EXISTS NodeTraitTraitInsert AFTER INSERT ON NodeTrait
  WHEN
    (SELECT 1 FROM Trait WHERE Name = NEW.Trait) IS NULL
  BEGIN
    SELECT RAISE(ABORT, 'Trait does not exist');
  END;

-- Edge.ParentNode -> Node.Name
CREATE TRIGGER IF NOT EXISTS EdgeParentNodeInsert AFTER INSERT ON Edge
  WHEN
    (SELECT 1 FROM Node WHERE Name = NEW.ParentNode) IS NULL
  BEGIN
    SELECT RAISE(ABORT, 'Dependency does not exist');
  END;

-- Edge.ChildNode -> Node.Name
CREATE TRIGGER IF NOT EXISTS EdgeChildNodeInsert AFTER INSERT ON Edge
  WHEN
    (SELECT 1 FROM Node WHERE Name = NEW.ChildNode) IS NULL
  BEGIN
    SELECT RAISE(ABORT, 'Node does not exist');
  END;

-- Parent.ParentNode -> Node.Name
CREATE TRIGGER IF NOT EXISTS ParentParentNodeInsert AFTER INSERT ON Parent
  WHEN
    (SELECT 1 FROM Node WHERE Name = NEW.ParentNode) IS NULL
  BEGIN
    SELECT RAISE(ABORT, 'Dependency does not exist');
  END;

-- Edge.ChildNode -> Node.Name
CREATE TRIGGER IF NOT EXISTS ParentChildNodeInsert AFTER INSERT ON Parent
  WHEN
    (SELECT 1 FROM Node WHERE Name = NEW.ChildNode) IS NULL
  BEGIN
    SELECT RAISE(ABORT, 'Node does not exist');
  END;

-- NodeFormula dimension check
CREATE TRIGGER IF NOT EXISTS NodeFormulaDimensionCheck AFTER INSERT ON
  NodeFormula
  WHEN
    (SELECT 1 FROM Node WHERE
    (NEW.XOrdinate >= 0 AND HasX IS NULL) OR
    (NEW.YOrdinate >= 0 AND HasY IS NULL) OR
    (NEW.ZOrdinate >= 0 AND HasZ IS NULL)) IS NOT NULL
  BEGIN
    SELECT RAISE(Abort, 'Formula uses dimension that does not exist');
  END;

-- Cascades
CREATE TRIGGER IF NOT EXISTS NodeUpdate
  AFTER
    UPDATE OF Name ON Node
  BEGIN
    UPDATE NodeTrait SET Node = NEW.Name WHERE Node = OLD.Name;
    UPDATE Edge SET ParentNode = NEW.Name WHERE ParentNode = OLD.Name;
    UPDATE Edge SET ChildNode = NEW.Name WHERE ChildNode = OLD.Name;
  END;

CREATE TRIGGER IF NOT EXISTS NodeDelete
  AFTER
    DELETE ON Node
  BEGIN
    DELETE FROM NodeTrait WHERE Node = OLD.Name;
    DELETE FROM Edge WHERE ParentNode = OLD.Name OR ChildNode = OLD.Name;
  END;

CREATE TRIGGER IF NOT EXISTS TraitUpdate
  AFTER
    UPDATE OF Name ON Trait
  BEGIN
    UPDATE NodeTrait SET Trait = NEW.Name WHERE Trait = OLD.Name;
  END;

CREATE TRIGGER IF NOT EXISTS TraitDelete
  AFTER
    DELETE ON Trait
  BEGIN
    DELETE FROM NodeTrait WHERE Trait = OLD.Name;
  END;

--
-- Other constraints
--
CREATE TRIGGER IF NOT EXISTS FormulaCheck AFTER INSERT ON Edge
  WHEN
    (SELECT 1 FROM NodeFormula WHERE Node = NEW.ChildNode AND
    Formula LIKE '%' || NEW.ParentNode || '%') IS NULL
  BEGIN
    SELECT RAISE(ABORT, 'Dependency is not mentioned in formulae');
  END;

--
-- Clear tables (in case they already existed)
--
DELETE FROM Node;
DELETE FROM Trait;
DELETE FROM Category;
DELETE FROM NodeTrait;
DELETE FROM NodeFormula;
DELETE FROM Edge;
DELETE FROM Parent;

--
-- Populate Trait
--
INSERT INTO Trait VALUES ('IS_S_NODE');
INSERT INTO Trait VALUES ('IS_D_NODE');
INSERT INTO Trait VALUES ('IS_C_NODE');
INSERT INTO Trait VALUES ('IS_R_NODE');
INSERT INTO Trait VALUES ('IS_F_NODE');
INSERT INTO Trait VALUES ('IS_O_NODE');
INSERT INTO Trait VALUES ('IS_P_NODE');
INSERT INTO Trait VALUES ('IS_GENERIC_STATIC');
INSERT INTO Trait VALUES ('IS_GENERIC_FORWARD');
INSERT INTO Trait VALUES ('IS_ODE_FORWARD');
INSERT INTO Trait VALUES ('IS_SDE_FORWARD');
INSERT INTO Trait VALUES ('IS_UNIFORM_VARIATE');
INSERT INTO Trait VALUES ('IS_GAUSSIAN_VARIATE');
INSERT INTO Trait VALUES ('IS_NORMAL_VARIATE');
INSERT INTO Trait VALUES ('IS_WIENER_INCREMENT');
INSERT INTO Trait VALUES ('IS_GAUSSIAN_LIKELIHOOD');
INSERT INTO Trait VALUES ('IS_NORMAL_LIKELIHOOD');
INSERT INTO Trait VALUES ('IS_LOG_NORMAL_LIKELIHOOD');
INSERT INTO Trait VALUES ('HAS_ZERO_MU');
INSERT INTO Trait VALUES ('HAS_UNIT_SIGMA');
INSERT INTO Trait VALUES ('HAS_COMMON_SIGMA');
INSERT INTO Trait VALUES ('HAS_GAUSSIAN_PRIOR');
INSERT INTO Trait VALUES ('HAS_NORMAL_PRIOR');
INSERT INTO Trait VALUES ('HAS_LOG_NORMAL_PRIOR');
INSERT INTO Trait VALUES ('HAS_UNIFORM_PRIOR');
INSERT INTO Trait VALUES ('HAS_GAMMA_PRIOR');
INSERT INTO Trait VALUES ('HAS_INVERSE_GAMMA_PRIOR');
INSERT INTO Trait VALUES ('HAS_CYCLIC_X_BOUNDARY');
INSERT INTO Trait VALUES ('HAS_CYCLIC_Y_BOUNDARY');
INSERT INTO Trait VALUES ('HAS_CYCLIC_Z_BOUNDARY');
INSERT INTO Trait VALUES ('HAS_EXPLICIT_X_BOUNDARY');
INSERT INTO Trait VALUES ('HAS_EXPLICIT_Y_BOUNDARY');
INSERT INTO Trait VALUES ('HAS_EXPLICIT_Z_BOUNDARY');

--
-- Populate category
--
INSERT INTO Category VALUES('Constant', '', 1);
INSERT INTO Category VALUES('Parameter', '', 2);
INSERT INTO Category VALUES('Forcing', '', 3);
INSERT INTO Category VALUES('Observation', '', 4);
INSERT INTO Category VALUES('Random variate', 'Representing pseudorandom variates required in the update of other variables.', 5);
INSERT INTO Category VALUES('Static variable', '.', 6);
INSERT INTO Category VALUES('Discrete-time variable', '', 7);
INSERT INTO Category VALUES('Continuous-time variable', '', 8);
INSERT INTO Category VALUES('Intermediate result', 'Representing intermediate evaluations which may be reused multiple times for convenience. Will be inlined.', 9);
